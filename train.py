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


def create_baseline_rewards(env: Chargax, num_iterations=100):
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

    # env = eqx.tree_at(lambda x: x.full_info_dict, env, True)
    baseline_rewards = {}
    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    obs, env_state = env.reset(reset_key)
    for method in ["random_actions", "maximum_actions"]:
        baseline_rewards[method] = {
            "episode_rewards": [],
            "profit": []
        }
        if method == "random_actions":
            step_env = step_env_random
        elif method == "maximum_actions":
            step_env = step_env_max
        for _ in range(num_iterations):
            (rng, obs, env_state, done, episode_reward), info = jax.lax.scan(
                step_env, 
                (rng, obs, env_state, False, 0.0), 
                length=env.episode_length
            )
            baseline_rewards[method]["episode_rewards"].append(episode_reward)
            baseline_rewards[method]["profit"].append(info["logging_data"]["profit"][-1])

    baseline_rewards = {
        k: {
            "episode_rewards": np.mean(v["episode_rewards"]),
            "profit": np.mean(v["profit"])
        } for k, v in baseline_rewards.items()
    }
    return baseline_rewards

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--seed", type=int, default=42)
    argument_parser.add_argument("--user_profiles", type=str, choices=["highway", "residential", "workplace", "shopping"], required=True)
    argument_parser.add_argument("--arrival_frequency", type=str, choices=["low", "medium", "high"], required=True)
    argument_parser.add_argument("--groupname", type=str, default=None)
    args, extra_args = argument_parser.parse_known_args()

    # Convert extra_args to a dictionary. we assume that they set environment parameters.
    env_parameters = {}
    for i in range(0, len(extra_args), 2):
        key = extra_args[i].lstrip('--')
        # check if the value is a float or an int
        if "." in extra_args[i + 1]:
            try:
                value = float(extra_args[i + 1])
            except ValueError:
                value = extra_args[i + 1]
        else:
            try:
                value = int(extra_args[i + 1])
            except ValueError:
                value = extra_args[i + 1]
        # if its a False or True string, convert it to a boolean
        if value == "False":
            value = False
        elif value == "True":
            value = True
        env_parameters[key] = value

    env = Chargax(
        elec_grid_buy_price=get_electricity_prices(),
        elec_grid_sell_price=get_electricity_prices(),
        user_profiles=args.user_profiles,
        arrival_frequency=args.arrival_frequency,
        **env_parameters
    )

    baselines = create_baseline_rewards(env)
    random_trainer_train_fn, config = build_ppo_trainer(
        env, 
        config_params={
            "total_timesteps": 10000000,
            "seed": args.seed
        },
        baselines=baselines
    )#, {"num_envs": 1, "total_timesteps": 1000})

    start_time = time.time()
    print("Starting JAX compilation...")
    random_trainer_train_fn = jax.jit(random_trainer_train_fn).lower().compile()
    print(
        f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
    )
    groupname = args.groupname if args.groupname else args.user_profiles + "_" + args.arrival_frequency
    c_time = time.time()
    wandb.init(project="chargax", entity="FelixAndKoen", config=config.__dict__, group=groupname)
    trained_runner_state, train_rewards = random_trainer_train_fn()
    print("Training finished")
    print(f"Training took {time.time() - c_time:.2f} seconds")

    trained_agent = trained_runner_state[0]
    key = trained_runner_state[-1]

    # env = Chargax(
    #     elec_grid_buy_price=get_electricity_prices(),
    #     elec_grid_sell_price=get_electricity_prices() - 0.1,
    #     full_info_dict=True
    # )

    wandb.finish()

    # episode_reward, infos = eval_func(trained_agent, key)
    # breakpoint()
    # print(f"Episode reward: {episode_reward}")
    # log_info(infos)