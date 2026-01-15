import argparse
import logging

import jax
import jaxnasium as jym
import matplotlib.pyplot as plt
import numpy as np
from jaxnasium.algorithms import PPO
from jaxtyping import PRNGKeyArray

from chargax import (
    Chargax,
    ChargingStation,
    EnvState,
    StationEVSE,
    StationSplitter,
    get_electricity_prices,
)
from chargax.baselines import LPAgent, MaxCharge, Random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Training configuration
RL_NUM_TRAINING_TIMESTEPS = 1_000_000
NUM_EVAL_EPISODES = 25
NUM_COMPARISON_EPISODES = 3

# Environment configuration
GRID_BUY_PRICE = 0.21
GRID_SELL_MARGIN = 0.02
CUSTOMER_SELL_PRICE = 0.78


class SimpleEnv(ChargingStation):
    def __init__(self):
        charger = StationSplitter(
            connections=[
                StationEVSE(voltage_rated=600, current_max=500, connections=[0, 1]),
            ],
            group_capacity_max_kw=300,
        )
        self.charger_layout = StationSplitter(
            connections=[charger], group_capacity_max_kw=250
        )


def create_env() -> jym.LogWrapper:
    """Create and wrap the charging environment."""
    base_prices = get_electricity_prices("2023_NL")
    env = Chargax(
        elec_grid_buy_price=np.ones_like(base_prices) * GRID_BUY_PRICE,
        elec_grid_sell_price=np.ones_like(base_prices)
        * (GRID_BUY_PRICE - GRID_SELL_MARGIN),
        elec_customer_sell_price=CUSTOMER_SELL_PRICE,
        station=SimpleEnv(),
        user_profiles="highway",
    )
    return jym.LogWrapper(env)


def create_rl_agent() -> PPO:
    """Create a PPO agent with default hyperparameters."""
    return PPO(
        num_steps=300,
        num_envs=8,
        total_timesteps=RL_NUM_TRAINING_TIMESTEPS,
        learning_rate=2.5e-4,
        anneal_learning_rate=True,
        normalize_rewards=False,
        normalize_observations=True,
    )


def run_episode(
    agent, env: Chargax, rng: PRNGKeyArray, init_state: EnvState, init_obs
) -> np.ndarray:
    """Run a single episode and return rewards at each step."""

    def step_fn(carry, _):
        seed, state, obs = carry
        this_step_key, next_key = jax.random.split(seed)
        if agent.__class__.__name__ == "PPO":
            action = agent.get_action(
                this_step_key, observation=obs, state=agent.state, deterministic=True
            )
        else:
            action = agent.get_action(this_step_key, env_state=state, observation=obs)
        timestep, new_state = env.step(this_step_key, state, action)
        new_obs, *_ = timestep
        return (next_key, new_state, new_obs), new_state

    _, states = jax.lax.scan(
        step_fn, (rng, init_state, init_obs), None, length=env.max_episode_steps
    )
    return states


def evaluate_agents(
    rng: PRNGKeyArray, env: Chargax, num_episodes: int = 25, skip_train: bool = False
):
    """train and evaluate rl agent against baselines and plot results."""

    agent = create_rl_agent()
    if skip_train:
        logger.info("Skipping training, initializing agent state...")
        trained_rl_agent = agent.init_state(rng, env)
    else:
        logger.info(
            f"Training PPO agent for {RL_NUM_TRAINING_TIMESTEPS:,} timesteps..."
        )
        trained_rl_agent = agent.train(rng, env)

    # Compare against baselines
    logger.info("Comparing trained agent against baselines...")
    agents = {
        "RL Agent": trained_rl_agent,
        "MaxCharge": MaxCharge(env),
        "Random": Random(env),
        "LP": LPAgent(env, rollout_fn=run_episode, oracle_mode=False),
        "LP Oracle": LPAgent(env, rollout_fn=run_episode, oracle_mode=True),
    }
    colors = {
        "RL Agent": "blue",
        "MaxCharge": "green",
        "Random": "red",
        "LP": "orange",
        "LP Oracle": "purple",
    }

    results = {name: [] for name in agents}

    for episode in range(num_episodes):
        rng, episode_rng = jax.random.split(rng)
        obs, env_state = env.reset(episode_rng)

        for name, agent in agents.items():
            states = run_episode(agent, env, episode_rng, env_state, obs)
            rewards = states.env_state.profit[:-1]
            total_reward = np.sum(rewards)
            logger.info(f"{name} Episode {episode + 1} reward: {total_reward:.2f}")
            results[name].append(rewards)

    # Plot cumulative rewards
    plt.figure(figsize=(10, 6))
    for name, reward_list in results.items():
        mean_rewards = np.mean(np.array(reward_list), axis=0)
        std_rewards = np.std(np.array(reward_list), axis=0)
        plt.plot(mean_rewards, label=name, color=colors[name])
        plt.fill_between(
            range(len(mean_rewards)),
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            color=colors[name],
            alpha=0.2,
        )

    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.title("Agent Comparison: Cumulative Reward")
    plt.legend()
    plt.savefig("episode_cumulative_reward.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL agent with baselines"
    )
    parser.add_argument(
        "-s",
        "--skip-train",
        action="store_true",
        help="Skip training and only initialize agent state",
    )
    args = parser.parse_args()

    rng = jax.random.PRNGKey(42)
    env = create_env()

    evaluate_agents(
        rng, env, num_episodes=NUM_COMPARISON_EPISODES, skip_train=args.skip_train
    )


if __name__ == "__main__":
    main()
