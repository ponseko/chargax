from chargax import (
    Chargax,
    get_scenario,
    get_electricity_prices,
    get_car_data,
    pretty_print_charger_group,
    build_random_trainer,
    build_ppo_trainer
)

import jax 
import jax.numpy as jnp
import time
import wandb
from dataclasses import replace
import chex
from typing import Literal
import numpy as np
import equinox as eqx
import argparse

env = Chargax(
    elec_grid_buy_price=get_electricity_prices(),
    elec_grid_sell_price=get_electricity_prices() - 0.03,
)

random_trainer_train_fn, config = build_ppo_trainer(env)#, {"num_envs": 1, "total_timesteps": 1000})

def step_env_random(carry, _):
    rng, env_state = carry
    rng, action_key, step_key = jax.random.split(rng, 3)
    action = env.action_space.sample(action_key)
    (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
    return (rng, env_state), _

def run_for_number_of_steps(key, num_steps=100000):
    rng = jax.random.PRNGKey(key)
    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)
    runner_state = (rng, env_state)
    (rng, env_state), _ = jax.lax.scan(step_env_random, runner_state, length=num_steps)
    return rng

start_time = time.time()
print("Starting JAX compilation...")
run_random_episode = jax.jit(run_for_number_of_steps).lower(0).compile()
print(
    f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
)
c_time = time.time()

for _ in range(1):
    run_random_episode(0)
print("Time for random:", time.time() - c_time)