import argparse

import jax
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import DQN, PPO, SAC

from chargax import Chargax, get_electricity_prices  # noqa: E402

if __name__ == "__main__":
    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
    )
    env = jym.LogWrapper(env)
    rng = jax.random.PRNGKey(42)

    # RL Training with PPO
    total_timesteps = 1_000_000
    agent = PPO(  # Not optimized, just a simple example
        num_steps=300,
        num_envs=8,
        total_timesteps=total_timesteps,
        learning_rate=2.5e-4,
        anneal_learning_rate=True,
        normalize_rewards=False,
        normalize_observations=True,  # Important
    )

    agent = agent.train(rng, env)

    results = agent.evaluate(rng, env, num_eval_episodes=25)
    print(f"Average reward over 25 evaluation episodes: {np.mean(results)}")
