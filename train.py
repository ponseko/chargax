from algorithms.random import build_random_trainer
from environment import Chargax
from scenarios.charger_topologies import create_uniform_topology
from scenarios.ev_arivals import interpolate_arrival_data
from scenarios.scenario_data import office_distribution_means
from util.helpers import pretty_print_charger_group

import jax 
import jax.numpy as jnp
import time

if __name__ == "__main__":
    charger_topology = create_uniform_topology(10, 2)
    arrival_distributions = interpolate_arrival_data(
        list(office_distribution_means.values()), 5, 1
    )
    env = Chargax(
        charger_topology=charger_topology,
        arrival_distributions=arrival_distributions
    )
    # obs, env_state = env.reset(jax.random.key(0))
    # for i in range(5):
    #     random_action = env.action_space.sample(jax.random.PRNGKey(i))
    #     (obs, reward, terminated, truncated, info), env_state = env.step(jax.random.key(i), env_state, random_action)

    # raise NotImplementedError("Please implement the training loop")

    random_trainer_train_fn = build_random_trainer(env)


    start_time = time.time()
    print("Starting JAX compilation...")
    random_trainer_train_fn = jax.jit(random_trainer_train_fn).lower().compile()
    print(
        f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
    )
    trained_state, train_rewards = random_trainer_train_fn()
    print("Training finished")

    print(jnp.sum(train_rewards))

    
    # obs, env_state = env.reset(jax.random.key(0))
    # pretty_print_charger_group(charger_topology, env_state.chargers)

    # for i in range(12):
    #     (obs, reward, terminated, truncated, info), env_state = env.step(jax.random.key(0), env_state, 0)

    #     pretty_print_charger_group(charger_topology, env_state.chargers)
