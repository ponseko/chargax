"""Integration tests for passive nodes in the Chargax environment."""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import pytest

from chargax._station_layout import ChargingStation, PassiveFlexNode, PassiveForcedNode
from chargax.chargax import Chargax, EnvState
from tests.test_station_layout_distribute import _evse_with_kw


def _station_with_passive(load_profile, *, flex: bool = False) -> ChargingStation:
    passive_cls = PassiveFlexNode if flex else PassiveForcedNode
    return ChargingStation(
        max_kw_throughput=500.0,
        efficiency=1.0,
        connections=[
            passive_cls(load_profile=load_profile),
            _evse_with_kw([0.0], shared_max_kw=100.0),
        ],
    )


def _env_for_station(station: ChargingStation) -> Chargax:
    return Chargax(
        station=station,
        get_num_cars_arriving=lambda k, s: jnp.int32(0),
        get_new_cars_arriving=lambda k, s: station.evses_flat,
        get_grid_buy_price=lambda s: 0.1,
        get_grid_sell_price=lambda s: 0.09,
    )


def test_set_passive_throughputs_constant_profile():
    station = _station_with_passive(80.0)
    env = _env_for_station(station)
    state = EnvState(
        datetime=jdt.to_datetime("2024-06-01"),
        grid=station,
        elec_customer_sell_price=0.75,
    )
    state = env.set_passive_throughputs(state)
    assert float(state.grid.passives[0].throughput_now_kw) == pytest.approx(80.0)


def test_set_passive_throughputs_callable_profile():
    station = _station_with_passive(lambda s: s.timestep * 5.0)
    env = _env_for_station(station)
    state = EnvState(
        datetime=jdt.to_datetime("2024-06-01"),
        grid=station,
        elec_customer_sell_price=0.75,
        timestep=4,
    )
    state = env.set_passive_throughputs(state)
    assert float(state.grid.passives[0].throughput_now_kw) == pytest.approx(20.0)


def test_reset_applies_passive_load():
    station = _station_with_passive(25.0)
    env = _env_for_station(station)
    _, state = env.reset_env(jax.random.PRNGKey(0))
    assert float(state.grid.passives[0].throughput_now_kw) == pytest.approx(25.0)


def test_set_passive_throughputs_applies_to_flex_node_too():
    station = _station_with_passive(-40.0, flex=True)
    env = _env_for_station(station)
    state = EnvState(
        datetime=jdt.to_datetime("2024-06-01"),
        grid=station,
        elec_customer_sell_price=0.75,
    )
    state = env.set_passive_throughputs(state)
    assert isinstance(state.grid.passives[0], PassiveFlexNode)
    assert float(state.grid.passives[0].throughput_now_kw) == pytest.approx(-40.0)
