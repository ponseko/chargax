from chargax import (
    Chargax,
    interpolate_arrival_data,
    office_distribution_means
)
import jax
import jax.numpy as jnp

def check_charger_state_over_capacity(charger_state, splitters):
    for splitter in splitters:
        assert splitter.total_power_throughput_kw(charger_state) <= (splitter.group_capacity_max_kw + 1e-5)

def test_normalization():
    arrival_distributions = interpolate_arrival_data(
        list(office_distribution_means.values()), 5, 1
    )
    env = Chargax(
        ev_arrival_data_means=arrival_distributions[0],
        ev_arrival_data_stds=arrival_distributions[1],
    )
    obs, _env_state = env.reset(jax.random.key(0))

    action_high = env.action_space.nvec[0]
    action_low = env.action_space.start

    max_actions = jnp.ones(env.action_space.n) * action_high + action_low
    _, env_state = env.step(jax.random.key(0), _env_state, max_actions)
    check_charger_state_over_capacity(env_state.chargers_state, env.station.splitters)

    min_actions = jnp.zeros(env.action_space.n)
    _, env_state = env.step(jax.random.key(0), _env_state, min_actions)
    check_charger_state_over_capacity(env_state.chargers_state, env.station.splitters)

    # try 100 random actions
    for i in range(100):
        random_actions = env.action_space.sample(jax.random.key(i))
        _, env_state = env.step(jax.random.key(0), _env_state, random_actions)
        check_charger_state_over_capacity(env_state.chargers_state, env.station.splitters)

    

