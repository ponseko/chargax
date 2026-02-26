import jax
import jaxnasium as jym
import numpy as np
from jaxnasium.algorithms import DQN, PPO, SAC

from chargax import Chargax, ChargingStation

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)

    # Initialize a default charging station environment from template:
    charging_station = ChargingStation.init_default_station()

    # Create the environment
    env = Chargax(station=charging_station)
    env = jym.LogWrapper(env)

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
