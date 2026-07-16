"""
Unified experiment runner for Chargax.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb
import pandas as pd

import jaxnasium as jym
from jaxnasium.algorithms import DQN, PPO, SAC
from jaxtyping import PRNGKeyArray

from chargax import Chargax
from chargax.baselines import MaxCharge, Random, FixedRateCharge
from chargax.scenarios.scenarios_config import build_scenario, list_scenarios


# --------------------------------------------------------------------------
# 1. Agent registry
# --------------------------------------------------------------------------

RL_AGENTS: dict[str, Callable[..., Any]] = {
    "ppo": PPO,
    "dqn": DQN,
    "sac": SAC,
}

BASELINE_AGENTS: dict[str, Callable[..., Any]] = {
    "max_charge": MaxCharge,
    "fixed_rate_charge": FixedRateCharge,
    "random": Random,
}


def is_rl_agent(agent_name: str) -> bool:
    return agent_name in RL_AGENTS

# --------------------------------------------------------------------------
# 2. Run configuration
# --------------------------------------------------------------------------

@dataclass
class RunConfig:
    scenario_name: str
    agent_name: str
    agent_kwargs: dict = field(default_factory=dict)    
    train_kwargs: dict = field(default_factory=dict)    
    scenario_overrides: dict = field(default_factory=dict)  
    num_eval_episodes: int = 25
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "chargax"
    wandb_group: str | None = None  # e.g. the sweep name, to group runs in the UI

    @property
    def run_name(self) -> str:
        "Creates the run name based on the scenario, agent and possible overrides (grid testing)"
        override_str = "_".join(f"{k}={v}" for k, v in self.scenario_overrides.items())
        parts = [self.scenario_name, self.agent_name]
        if override_str:
            parts.append(override_str)
        return "-".join(parts)

# --------------------------------------------------------------------------
# 4. Evaluation helper — normalizes RL-agent vs baseline-agent evaluate() output
# --------------------------------------------------------------------------

def evaluate_rl_agent(agent, env, key, num_eval_episodes):
    rewards = agent.evaluate(
        key,
        env,
        num_eval_episodes=num_eval_episodes,
    )

    rewards = np.asarray(rewards)

    # If evaluate returns one reward per episode
    if rewards.ndim == 1:
        avg_reward = float(rewards.mean())
    # If it returns reward per episode per environment or timestep
    else:
        avg_reward = float(rewards.mean())

    return {
        "avg_reward": avg_reward,
    }


def evaluate_baseline(agent, key, num_eval_episodes):
    rewards, profits = agent.evaluate(key, num_eval_episodes)

    rewards = np.asarray(rewards)
    profits = np.asarray(profits)

    return {
        "avg_reward": float(np.sum(rewards, axis=1).mean()),
        "avg_profit": float(np.sum(profits, axis=1).mean()),
    }

# --------------------------------------------------------------------------
# 5. Single run
# --------------------------------------------------------------------------

def run_single(cfg: RunConfig) -> dict:
    """Builds the scenario + agent described by `cfg`, trains if needed,
    evaluates, optionally logs to wandb, and returns a flat results dict.
    """
    rng = jax.random.PRNGKey(cfg.seed)

    env = build_scenario(cfg.scenario_name, cfg.scenario_overrides)
    env = jym.LogWrapper(env)

    wandb_run = None
    if cfg.use_wandb:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.run_name,
            config={
                "scenario": cfg.scenario_name,
                "agent": cfg.agent_name,
                **cfg.agent_kwargs,
                **cfg.train_kwargs,
                **{f"override__{k}": v for k, v in cfg.scenario_overrides.items()},
            },
            reinit=True,
        )

    start = time.time()

    if is_rl_agent(cfg.agent_name):
        agent_cls = RL_AGENTS[cfg.agent_name]
        log_function = (lambda *a, **kw: wandb.log(*a, **kw)) if cfg.use_wandb else None
        agent = agent_cls(
            **cfg.train_kwargs,
            log_function=log_function,
            log_interval=1,  # must always be a positive int; log_function=None disables logging
        )
        agent = agent.train(rng, env)
        summary = evaluate_rl_agent(agent, env, rng, cfg.num_eval_episodes)
    elif cfg.agent_name in BASELINE_AGENTS:
        agent_cls = BASELINE_AGENTS[cfg.agent_name]
        agent = agent_cls(env, **cfg.agent_kwargs)
        summary = evaluate_baseline(agent, rng, cfg.num_eval_episodes)
    else:
        raise ValueError(
            f"Unknown agent_name '{cfg.agent_name}'. "
            f"Known RL agents: {list(RL_AGENTS)}. Known baselines: {list(BASELINE_AGENTS)}."
        )

    elapsed = time.time() - start
    summary.update(
        {
            "scenario": cfg.scenario_name,
            "agent": cfg.agent_name,
            "run_name": cfg.run_name,
            "elapsed_sec": elapsed,
            **{f"override__{k}": v for k, v in cfg.scenario_overrides.items()},
        }
    )

    if wandb_run is not None:
        wandb.log({f"final/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb_run.finish()

    print(f"[{cfg.run_name}] done in {elapsed:.1f}s -> {summary}")
    return summary


# --------------------------------------------------------------------------
# 6. Sweep over scenarios x agents x objective-alpha grid
# --------------------------------------------------------------------------

def build_sweep_configs(
    scenario_names: list[str],
    agent_names: list[str],
    alpha_grid: dict[str, list] | None = None,
    agent_kwargs_by_name: dict[str, dict] | None = None,
    train_kwargs_by_name: dict[str, dict] | None = None,
    num_eval_episodes: int = 25,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "chargax",
    wandb_group: str | None = None,
) -> list[RunConfig]:
    """Builds one RunConfig per (scenario, agent, alpha-combination).
    """
    agent_kwargs_by_name = agent_kwargs_by_name or {}
    train_kwargs_by_name = train_kwargs_by_name or {}
    alpha_grid = alpha_grid or {}

    if alpha_grid:
        keys = list(alpha_grid.keys())
        value_combinations = list(itertools.product(*alpha_grid.values()))
        override_dicts = [dict(zip(keys, combo)) for combo in value_combinations]
    else:
        override_dicts = [{}]

    configs = []
    for scenario_name, agent_name, overrides in itertools.product(
        scenario_names, agent_names, override_dicts
    ):
        configs.append(
            RunConfig(
                scenario_name=scenario_name,
                agent_name=agent_name,
                agent_kwargs=agent_kwargs_by_name.get(agent_name, {}),
                train_kwargs=train_kwargs_by_name.get(agent_name, {}) if is_rl_agent(agent_name) else {},
                scenario_overrides=overrides,
                num_eval_episodes=num_eval_episodes,
                seed=seed,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
            )
        )
    return configs


def run_sweep(configs: list[RunConfig]) -> list[dict]:
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n=== Run {i + 1}/{len(configs)}: {cfg.run_name} ===")
        results.append(run_single(cfg))
    return results

# --------------------------------------------------------------------------
# 7. Example usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    USE_WANDB = True  # flip this one flag to toggle logging everywhere
    total_timesteps = 500_000

    print("Available scenarios:", list_scenarios())

    configs = build_sweep_configs(
        scenario_names=["home"],   # whichever scenario keys you have
        agent_names=["dqn", "sac", "ppo", "max_charge", "fixed_rate_charge", "random"], # "dqn", "sac",
        # alpha_grid={
        #     "capacity_exceeded_alpha": [0],
        #     "charged_satisfaction_alpha": [0],
        # },
        agent_kwargs_by_name={
            "fixed_rate_charge": {"charge_rate": 0.5},
        },
        train_kwargs_by_name={
            "ppo": {
                "num_steps": 300,
                "num_envs": 8,
                "total_timesteps": total_timesteps,
                "learning_rate": 2.5e-4,
                "anneal_learning_rate": True,
                "normalize_rewards": False,
                "normalize_observations": True,  # Important
            },
            "dqn": {
                "num_envs": 8,
                "total_timesteps": total_timesteps,
                "learning_rate": 2.5e-4,
                "anneal_learning_rate": True,
                "normalize_rewards": False,
                "normalize_observations": True,
            },
            "sac": {
                "num_envs": 8,
                "total_timesteps": total_timesteps,
                "learning_rate_actor": 2.5e-4,
                "learning_rate_critics": 2.5e-4,
                "anneal_learning_rate_actor": True,
                "anneal_learning_rate_critics": True,
                "normalize_rewards": False,
                "normalize_observations": True,
            },
        },
        num_eval_episodes=25,
        use_wandb=USE_WANDB,
        seed = 43,
        wandb_project="chargax",
        wandb_group="alpha_sweep_v1",
    )

    print(f"\nRunning {len(configs)} experiments "
          f"({'with' if USE_WANDB else 'without'} wandb logging)...\n")
    

    # Creates a dataframe and and saves this as a csv file. 
    results = run_sweep(configs)
    table = pd.DataFrame(results)

    print("\n=== Summary ===")
    print(table)

    table.to_csv("sweep_results.csv", index=False)
    print("\nSaved results to sweep_results.csv")


# import jax
# import jax.numpy as jnp
# import jaxnasium as jym
# import numpy as np
# from jaxnasium.algorithms import DQN, PPO, SAC
# from jaxtyping import Array, PRNGKeyArray
# import wandb

# from chargax import EVSE, Chargax, ChargingStation
# from chargax.baselines import MaxCharge, Random, FixedRateCharge
# from chargax.scenarios.scenarios_config import build_scenario, list_scenarios
# from wandb_logger import wandb_logger


# if __name__ == "__main__":
#     rng = jax.random.PRNGKey(42)

#     total_timesteps = 500000
#     learning_rate = 2.5e-4
#     num_steps = 300
#     num_envs = 12

#     wandb.init(
#         project="chargax",
#         config={
#             "num_steps": num_steps,
#             "num_envs": num_envs,
#             "total_timesteps": total_timesteps,
#             "learning_rate": learning_rate,
#         },
#     )

#     env = build_scenario("residential")
#     env = jym.LogWrapper(env)

#     # RL Training with PPO
#     agent = PPO(  # Not optimized, just a simple example
#         num_steps=num_steps,
#         num_envs=num_envs,
#         total_timesteps=total_timesteps,
#         learning_rate=learning_rate,
#         anneal_learning_rate=True,
#         normalize_rewards=False,
#         normalize_observations=True,  # Important
#         log_function=wandb_logger,   # custom wandb callback
#         log_interval=1,
#         gamma= 0.99,
#         gae_lambda = 0.95,
#         max_grad_norm = 100.0,
#         clip_coef= 0.2,
#         clip_coef_vf= 10.0, # Depends on the reward scaling !,
#         ent_coef= 0.01,
#         vf_coef = 0.25,
#         num_minibatches = 4, # Number of mini-batches,
#         #update_epochs = 4, # K epochs to update the policy                        
#     )
#     agent = agent.train(rng, env)

#     results = agent.evaluate(rng, env, num_eval_episodes=25)

#     print(
#         f"PPO - Average reward over 25 evaluation episodes: {np.mean(results)}"
#     )

#     # Compare against baselines:
#     print("Evaluating baselines...")
#     rewards, profits = MaxCharge(env).evaluate(rng, num_eval_episodes=10)
#     print(
#         f"MaxCharge - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
#         f"MaxCharge - Average cumulative profit: {np.sum(profits, axis=1).mean():.2f}"
#     )
#     rewards, profits = FixedRateCharge(env, charge_rate=0.5).evaluate(rng, num_eval_episodes=10)
#     print(
#         f"50% Charge - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
#         f" 50% Charge - Average cumulative profit: {np.sum(profits, axis=1).mean():.2f}"
#     )
#     rewards, profits = Random(env).evaluate(rng, num_eval_episodes=10)
#     print(
#         f"Random - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
#         f"Random - Average cumulative profit: {np.sum(profits, axis=1).mean():.2f}"
#     )

#     wandb.finish()

  


   