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

def eval_func(train_state, rng):

    def step_env(carry, _):
        rng, obs, env_state, done, episode_reward = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        action_dist = train_state.actor(obs)
        action = action_dist.sample(seed=action_key)
        (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
        done = jnp.logical_or(terminated, truncated)
        episode_reward += reward
        return (rng, obs, env_state, done, episode_reward), info

    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)
    done = False
    episode_reward = 0.0
    runner_state = (rng, obs, env_state, done, episode_reward)
    runner_state, infos = jax.lax.scan(step_env, runner_state, length=env.episode_length)

    return runner_state[-1], infos

def log_info(info):
    wandb.init(project="chargax", entity="FelixAndKoen", tags=["eval"])

    def split_array_into_dict_of_singles(log_item):
        if isinstance(log_item, chex.Array) and log_item.size > 1:
            return {i: c_item for i, c_item in enumerate(log_item)}
        return log_item

    for step in range(len(info["day_of_year"])):
        log = jax.tree.map(lambda x: x[step], info)
        log = jax.tree.map(lambda x: x.astype(int) if x.dtype == bool else x, log) # convert bools to ints for wandb
        log = jax.tree.map(split_array_into_dict_of_singles, log)
        wandb.log(log)

    wandb.finish()


def create_baseline_rewards(env: Chargax, num_iterations=10):
    """ 
        Create a baseline for a random and max action agent
        averaged over num_iterations
    """
    
    def step_env_random(carry, _):
        rng, obs, env_state, done, episode_reward = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        action = env.action_space.sample(action_key)
        (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
        episode_reward += reward
        done = jnp.logical_or(terminated, truncated)
        return (rng, obs, env_state, done, episode_reward), info

    def step_env_max(carry, _):
        rng, obs, env_state, done, episode_reward = carry
        rng, action_key, step_key = jax.random.split(rng, 3)
        max_action = env.action_space.nvec.max()
        action = np.ones_like(env.action_space.nvec) * max_action
        if env.include_battery:
            action[-1] = 0.0 # battery action --> maximum discharge
        (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
        episode_reward += reward
        done = jnp.logical_or(terminated, truncated)
        return (rng, obs, env_state, done, episode_reward), info

    baseline_rewards = {}
    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)
    for method in ["random_actions", "maximum_actions"]:
        baseline_rewards[method] = []
        if method == "random_actions":
            step_env = step_env_random
        elif method == "maximum_actions":
            step_env = step_env_max
        for _ in range(num_iterations):
            (rng, obs, env_state, done, episode_reward), _ = jax.lax.scan(
                step_env, 
                (rng, obs, env_state, False, 0.0), 
                length=env.episode_length
            )
            baseline_rewards[method].append(episode_reward)

    baseline_rewards = {k: np.mean(v) for k, v in baseline_rewards.items()}
    return baseline_rewards



# elif method == "random":
#             action = env.action_space.sample(action_key)
#         elif method == "max":
#             max_action = env.action_space.nvec.max()
#             action = jnp.ones_like(env.action_space.nvec) * max_action



if __name__ == "__main__":
    arrival_distributions = get_scenario("office")
    car_data = get_car_data(dataset="eu")
    env = Chargax(
        ev_arrival_means_workdays=arrival_distributions,
        ev_arrival_means_non_workdays=arrival_distributions,
        elec_grid_buy_price=get_electricity_prices(),
        elec_grid_sell_price=get_electricity_prices() - 0.1,
    )

    baselines = create_baseline_rewards(env)
    random_trainer_train_fn = build_ppo_trainer(env, baselines=baselines)#, {"num_envs": 1, "total_timesteps": 1000})

    start_time = time.time()
    print("Starting JAX compilation...")
    random_trainer_train_fn = jax.jit(random_trainer_train_fn).lower().compile()
    print(
        f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
    )
    c_time = time.time()
    wandb.init(project="chargax", entity="FelixAndKoen")
    trained_runner_state, train_rewards = random_trainer_train_fn()
    print("Training finished")
    print(f"Training took {time.time() - c_time:.2f} seconds")

    trained_agent = trained_runner_state[0]
    key = trained_runner_state[-1]

    env = Chargax(
        ev_arrival_means_workdays=arrival_distributions,
        ev_arrival_means_non_workdays=arrival_distributions,
        elec_grid_buy_price=get_electricity_prices(),
        elec_grid_sell_price=get_electricity_prices() - 0.1,
        full_info_dict=True
    )

    wandb.finish()

    episode_reward, infos = eval_func(trained_agent, key)
    # breakpoint()
    print(f"Episode reward: {episode_reward}")
    log_info(infos)