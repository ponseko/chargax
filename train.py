from chargax import (
    Chargax,
    get_scenario,
    get_electricity_prices,
    pretty_print_charger_group,
    build_random_trainer,
    build_ppo_trainer
)

import jax 
import jax.numpy as jnp
import time
import wandb

if __name__ == "__main__":
    get_electricity_prices()
    arrival_distributions = get_scenario("office")
    env = Chargax(
        ev_arrival_means_workdays=arrival_distributions,
        ev_arrival_means_non_workdays=arrival_distributions,
    )

    random_trainer_train_fn = build_ppo_trainer(env)

    start_time = time.time()
    print("Starting JAX compilation...")
    random_trainer_train_fn = jax.jit(random_trainer_train_fn).lower().compile()
    print(
        f"JAX compilation finished in {(time.time() - start_time):.2f} seconds, starting training..."
    )
    c_time = time.time()
    wandb.init(project="chargax", entity="FelixAndKoen")
    trained_state, train_rewards = random_trainer_train_fn()
    print("Training finished")
    print(f"Training took {time.time() - c_time:.2f} seconds")
