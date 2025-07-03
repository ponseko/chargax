import time
from functools import partial

import jax
import jymkit as jym
import numpy as np
from jymkit.algorithms import PPO

from chargax import Chargax, get_electricity_prices  # noqa: E402

if __name__ == "__main__":
    env = Chargax(
        elec_grid_buy_price=get_electricity_prices("2023_NL"),
        elec_grid_sell_price=get_electricity_prices("2023_NL") - 0.02,
    )
    env = jym.LogWrapper(env)
    rng = jax.random.PRNGKey(42)

    # RL Training with PPO
    agent = PPO()

    s_time = time.time()
    print("Start JAX compilation...")
    train_fn = jax.jit(partial(agent.train, rng, env)).lower().compile()
    print("JAX compilation finished, took {:.2f} seconds".format(time.time() - s_time))
    agent: PPO = train_fn()

    results = agent.evaluate(rng, env, num_eval_episodes=25)
    print(f"Average reward over 25 evaluation episodes: {np.mean(results)}")
