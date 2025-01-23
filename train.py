from environment import Chargax
from scenarios.charger_topologies import create_uniform_topology
from scenarios.ev_arivals import interpolate_arrival_data
from scenarios.scenario_data import office_distribution_means
from util.helpers import pretty_print_charger_group

import jax 
import jax.numpy as jnp

if __name__ == "__main__":
    charger_topology, chargers = create_uniform_topology(10, 2)
    arrival_distributions = interpolate_arrival_data(
        list(office_distribution_means.values()), 5, 1
    )
    pretty_print_charger_group(charger_topology, chargers)
    env = Chargax(
        charger_topology=charger_topology,
        chargers=chargers,
        arrival_distributions=arrival_distributions
    )
    obs, env_state = env.reset(jax.random.key(0))

    for _ in range(12):

        (obs, reward, terminated, truncated, info), env_state = env.step(jax.random.key(0), env_state, 0)

        pretty_print_charger_group(env_state.grid_connection)

