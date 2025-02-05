from chargax import (
    Chargax,
    interpolate_arrival_data,
    office_distribution_means,
    pretty_print_charger_group,
    build_random_trainer,
    build_ppo_trainer
)

import jax 
import jax.numpy as jnp
import time

if __name__ == "__main__":
    arrival_distributions = interpolate_arrival_data(
        list(office_distribution_means.values()), 5, 1
    )
    env = Chargax(
        ev_arrival_data_means=arrival_distributions[0],
        ev_arrival_data_stds=arrival_distributions[1],
    )
    # obs, env_state = env.reset(jax.random.key(0))
    # for i in range(5):
    #     random_action = env.action_space.sample(jax.random.PRNGKey(i))
    #     (obs, reward, terminated, truncated, info), env_state = env.step(jax.random.key(i), env_state, random_action)

    # raise NotImplementedError("Please implement the training loop")

    random_trainer_train_fn = build_random_trainer(
        env
    )


    start_time = time.time()
    print("Starting JAX compilation...")
    random_trainer_train_fn = jax.jit(random_trainer_train_fn).lower().compile()
    print(
        f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
    )
    c_time = time.time()
    trained_state, train_rewards = random_trainer_train_fn()
    print("Training finished")
    print(f"Training took {time.time() - c_time:.2f} seconds")
