import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import DQN, PPO, SAC
from jaxtyping import Array, PRNGKeyArray
import wandb

from chargax import EVSE, Chargax, ChargingStation
from chargax.baselines import MaxCharge, Random
from chargax.scenarios._scenarios_config import build_scenario, list_scenarios
from chargax.scenarios._scenarios_station_layout import init_default_homecharger, init_default_station
from wandb_logger import wandb_logger


if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)

    total_timesteps = 1000000
    learning_rate = 2.5e-4
    num_steps = 300
    num_envs = 12

    wandb.init(
        project="chargax",
        config={
            "num_steps": num_steps,
            "num_envs": num_envs,
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
        },
    )

    charging_station = ChargingStation.init_default_homecharger()


    env = build_scenario("home")
    env = jym.LogWrapper(env)

    # RL Training with PPO
    agent = PPO(  # Not optimized, just a simple example
        num_steps=num_steps,
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        anneal_learning_rate=True,
        normalize_rewards=False,
        normalize_observations=True,  # Important
        log_function=wandb_logger,   # custom wandb callback
        log_interval=1,
        gamma= 0.99,
        gae_lambda = 0.95,
        max_grad_norm = 100.0,
        clip_coef= 0.2,
        clip_coef_vf= 10.0, # Depends on the reward scaling !,
        ent_coef= 0.01,
        vf_coef = 0.25,
        num_minibatches = 4, # Number of mini-batches,
        #update_epochs = 4, # K epochs to update the policy                        
    )
    agent = agent.train(rng, env)

    results = agent.evaluate(rng, env, num_eval_episodes=25)
    print(f"PPO - Average reward over 25 evaluation episodes: {np.mean(results)}")

    wandb.finish()

    # # Compare against baselines:
    # print("Evaluating baselines...")
    # rewards, profits = MaxCharge(env).evaluate(rng, num_eval_episodes=10)
    # print(
    #     f"MaxCharge - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}"
    # )
    # rewards, profits = Random(env).evaluate(rng, num_eval_episodes=10)
    # print(f"Random - Average cumulative reward: {np.sum(rewards, axis=1).mean():.2f}")


   