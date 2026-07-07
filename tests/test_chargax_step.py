import jax
import jax.numpy as jnp

from chargax import EVSE, Chargax, ChargingStation
from chargax.baselines import Random


def _test_env() -> Chargax:
    station = ChargingStation(
        max_kw_throughput=100.0,
        efficiency=1.0,
        connections=[EVSE(num_chargers=1, voltage=400.0, max_current=32.0)],
    )
    return Chargax(
        station=station,
        get_num_cars_arriving=lambda k, s: jnp.int32(0),
        get_new_cars_arriving=lambda k, s: station.evses_flat,
        get_grid_buy_price=lambda s: 0.1,
        get_grid_sell_price=lambda s: 0.09,
        simulation_length_days=7,
    )


def test_step_runs_with_sampled_action():
    env = _test_env()
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    action = env.sample_action(key)
    timestep, new_state = env.step(key, state, action)

    assert obs is not None
    assert new_state.timestep == state.timestep + 1
    assert timestep.observation is not None
    assert timestep.reward is not None


def test_random_get_action_can_be_consumed_by_step():
    env = _test_env()
    key = jax.random.PRNGKey(1)

    _, state = env.reset(key)
    action = Random(env).get_action(key)
    timestep, new_state = env.step(key, state, action)

    assert new_state.timestep == state.timestep + 1
    assert timestep.reward is not None
