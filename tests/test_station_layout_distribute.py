"""Tests for power distribution across the station layout tree."""

import jax.numpy as jnp
import pytest

from chargax._station_layout import (
    EVSE,
    ChargingStation,
    PassiveFlexNode,
    PassiveForcedNode,
    StationBattery,
    StationSplitter,
)


def _evse_with_kw(
    kws: list[float],
    *,
    voltage: float = 400.0,
    max_current: float = 1250.0,
    shared_max_kw: float | None = None,
) -> EVSE:
    """Build an EVSE with per-charger power (kW); positive=charge, negative=discharge."""
    n = len(kws)
    evse = EVSE(num_chargers=n, voltage=voltage, max_current=max_current)
    if shared_max_kw is not None:
        evse = evse.replace(max_kw_throughput=jnp.full(n, shared_max_kw))
    currents = jnp.array(kws) * 1000.0 / voltage
    return evse.replace(
        charger_current_now=currents,
        charger_is_car_connected=jnp.ones(n, dtype=bool),
    )


def _passive(throughput_kw: float) -> PassiveForcedNode:
    """A forced passive: always takes precedence, never scaled by distribute().
    Positive = uncontrollable load, negative = uncontrollable generation."""
    return PassiveForcedNode(load_profile=0.0).replace(throughput_now_kw=throughput_kw)


def _passive_flex(
    throughput_kw: float, *, max_kw: float | None = None
) -> PassiveFlexNode:
    """A flex passive: curtailed like a controllable flow when the station
    can't carry it in full. Positive = load, negative = generation."""
    node = PassiveFlexNode(load_profile=0.0).replace(throughput_now_kw=throughput_kw)
    if max_kw is not None:
        node = node.replace(max_kw_throughput=max_kw)
    return node


def _battery(throughput_kw: float, *, max_kw: float = 500.0) -> StationBattery:
    return StationBattery(
        capacity_kw=1000.0,
        max_kw_throughput=max_kw,
        efficiency=1.0,
    ).replace(throughput_now_kw=throughput_kw)


def _grid(*connections, max_kw: float = 1000.0) -> ChargingStation:
    return ChargingStation(
        max_kw_throughput=max_kw,
        efficiency=1.0,
        connections=list(connections),
    )


def _splitter(*connections, max_kw: float) -> StationSplitter:
    return StationSplitter(
        max_kw_throughput=max_kw,
        efficiency=1.0,
        connections=list(connections),
    )


def _kw(evse: EVSE) -> list[float]:
    return [float(p) for p in evse.power_output]


def _assert_no_exceeded(splitter: StationSplitter, tol: float = 1e-1):
    assert float(splitter.exceeded_power_all_children) <= tol


class TestOriginalSplitterBatteryBug:
    """Grid >> splitter: EVSE load + battery discharge must respect splitter limit."""

    def test_battery_discharge_capped_when_evse_scaled(self):
        evse = _evse_with_kw(
            [300.0], voltage=400.0, max_current=750.0, shared_max_kw=750.0
        )
        battery = _battery(-400.0)
        tree = _grid(_splitter(evse, battery, max_kw=200.0), max_kw=1000.0)

        result = tree.distribute()
        split = result.connections[0]
        evse_out = split.connections[0]
        bat_out = split.connections[1]

        assert float(evse_out.requested_power) == pytest.approx(200.0)
        assert float(bat_out.throughput_now_kw) == pytest.approx(-200.0)
        assert float(bat_out.supplied_power) == pytest.approx(200.0)
        _assert_no_exceeded(split)
        _assert_no_exceeded(result)


class TestEVSEInternalV2G:
    """Discharge on one port increases charging budget on another port (same EVSE)."""

    def test_v2g_boosts_sibling_charging(self):
        evse = _evse_with_kw([400.0, -100.0], shared_max_kw=500.0)
        out = evse.distribute(200.0)

        assert _kw(out)[0] == pytest.approx(300.0)
        assert _kw(out)[1] == pytest.approx(-100.0)

    def test_v2g_via_constrained_splitter(self):
        evse = _evse_with_kw([400.0, -100.0], shared_max_kw=500.0)
        tree = _grid(_splitter(evse, max_kw=200.0))
        result = tree.distribute()
        evse_out = result.connections[0].connections[0]

        assert _kw(evse_out)[0] == pytest.approx(300.0)
        assert _kw(evse_out)[1] == pytest.approx(-100.0)


class TestUnderCapacity:
    def test_no_scaling_when_within_limits(self):
        evse = _evse_with_kw(
            [50.0, 60.0],
            voltage=230.0,
            max_current=50.0,
            shared_max_kw=200.0,
        )
        tree = _grid(_splitter(evse, max_kw=200.0), max_kw=500.0)
        result = tree.distribute()
        evse_out = result.connections[0].connections[0]

        assert _kw(evse_out) == pytest.approx([50.0, 60.0])


class TestDemandOnlyScaling:
    def test_splitter_scales_oversubscribed_load(self):
        evse = _evse_with_kw(
            [150.0, 150.0],
            voltage=230.0,
            max_current=50.0,
            shared_max_kw=500.0,
        )
        tree = _grid(_splitter(evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        evse_out = result.connections[0].connections[0]

        assert float(evse_out.requested_power) == pytest.approx(200.0)
        _assert_no_exceeded(result.connections[0])


class TestBattery:
    def test_charging_scaled_to_allocation(self):
        bat = _battery(300.0)
        out = bat.distribute(100.0)
        assert float(out.throughput_now_kw) == pytest.approx(100.0)

    def test_discharge_capped_on_export_allocation(self):
        bat = _battery(-400.0)
        out = bat.distribute(-200.0)
        assert float(out.throughput_now_kw) == pytest.approx(-200.0)

    def test_discharge_unchanged_on_import_allocation(self):
        bat = _battery(-100.0)
        out = bat.distribute(200.0)
        assert float(out.throughput_now_kw) == pytest.approx(-100.0)


class TestNestedLayout:
    def test_inner_splitter_constrains_subtree(self):
        fast = _evse_with_kw(
            [200.0], voltage=600.0, max_current=500.0, shared_max_kw=600.0
        )
        slow = _evse_with_kw(
            [40.0], voltage=230.0, max_current=50.0, shared_max_kw=50.0
        )
        battery = _battery(-300.0, max_kw=500.0)
        inner = _splitter(fast, slow, battery, max_kw=150.0)
        tree = _grid(inner, max_kw=1000.0)

        result = tree.distribute()
        inner_out = result.connections[0]

        assert float(inner_out.requested_power) <= 150.0 + 1e-3
        assert float(inner_out.supplied_power) <= 150.0 + 1e-3
        _assert_no_exceeded(inner_out)
        _assert_no_exceeded(result)

    def test_default_station_distribute_runs(self):
        tree = ChargingStation.init_default_station()
        result = tree.distribute()
        assert result.max_kw_throughput == pytest.approx(200.0)
        _assert_no_exceeded(result, tol=1e-3)


class TestMultipleSiblings:
    def test_two_evses_share_splitter_capacity(self):
        evse_a = _evse_with_kw(
            [200.0], voltage=400.0, max_current=500.0, shared_max_kw=500.0
        )
        evse_b = _evse_with_kw(
            [200.0], voltage=400.0, max_current=500.0, shared_max_kw=500.0
        )
        tree = _grid(_splitter(evse_a, evse_b, max_kw=250.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        total = float(split.requested_power)
        assert total == pytest.approx(250.0)
        _assert_no_exceeded(split)

    def test_evse_and_battery_sibling_supply_capped_to_node_limit(self):
        evse = _evse_with_kw([100.0], shared_max_kw=500.0)
        battery = _battery(-250.0)
        tree = _grid(_splitter(evse, battery, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]
        bat_out = split.connections[1]

        assert float(split.requested_power) == pytest.approx(100.0)
        assert float(split.supplied_power) == pytest.approx(200.0)
        assert float(bat_out.throughput_now_kw) == pytest.approx(-200.0)
        _assert_no_exceeded(split)


class TestEVSEExportCap:
    def test_v2g_capped_when_branch_is_export_limited(self):
        evse = _evse_with_kw([-200.0], voltage=400.0, max_current=500.0)
        out = evse.distribute(-200.0)
        assert float(out.supplied_power) == pytest.approx(200.0)


class TestPassiveForcedNode:
    def test_passive_load_reduces_controllable_budget(self):
        passive = _passive(80.0)
        evse = _evse_with_kw([200.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert float(split.connections[0].throughput_now_kw) == pytest.approx(80.0)
        assert float(split.connections[1].requested_power) == pytest.approx(120.0)

    def test_passive_generation_not_scaled(self):
        passive = _passive(-60.0)
        evse = _evse_with_kw([150.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert float(split.connections[0].throughput_now_kw) == pytest.approx(-60.0)
        assert float(split.connections[1].requested_power) == pytest.approx(150.0)

    def test_passive_unchanged_when_controllable_supply_capped(self):
        passive = _passive(50.0)
        evse = _evse_with_kw([300.0], shared_max_kw=500.0)
        battery = _battery(-400.0)
        tree = _grid(_splitter(passive, evse, battery, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert float(split.connections[0].throughput_now_kw) == pytest.approx(50.0)
        assert float(split.connections[1].requested_power) == pytest.approx(200.0)
        assert float(split.connections[2].throughput_now_kw) == pytest.approx(-250.0)


class TestPassivesFlat:
    def test_homogeneous_forced_passives_concatenate(self):
        tree = _grid(_passive(10.0), _passive(20.0), max_kw=500.0)
        flat = tree.passives_flat
        assert float(flat.throughput_now_kw[0]) == pytest.approx(10.0)
        assert float(flat.throughput_now_kw[1]) == pytest.approx(20.0)

    def test_homogeneous_flex_passives_concatenate(self):
        tree = _grid(_passive_flex(10.0), _passive_flex(20.0), max_kw=500.0)
        flat = tree.passives_flat
        assert float(flat.throughput_now_kw[0]) == pytest.approx(10.0)
        assert float(flat.throughput_now_kw[1]) == pytest.approx(20.0)

    def test_mixed_passive_types_raise_with_helpful_message(self):
        tree = _grid(_passive(10.0), _passive_flex(-20.0), max_kw=500.0)
        with pytest.raises(ValueError, match="passives_flat\\(\\) cannot concatenate"):
            _ = tree.passives_flat
        with pytest.raises(ValueError, match="Use the passives property instead"):
            _ = tree.passives_flat


class TestGridSurplusAccounting:
    """Discharge still counts as surplus upstream even when inner node caps export."""

    def test_grid_sees_net_after_inner_cap(self):
        evse = _evse_with_kw(
            [300.0], voltage=400.0, max_current=750.0, shared_max_kw=750.0
        )
        battery = _battery(-400.0)
        tree = _grid(_splitter(evse, battery, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert float(split.requested_power - split.supplied_power) == pytest.approx(
            0.0, abs=1e-3
        )


class TestUpstreamExportBudgetPropagation:
    """Regression: a battery below a splitter that is *larger* than the grid
    connection could discharge its full rating, net-exporting far more than the
    grid connection could carry, because the export limit was only enforced by
    the local splitter and never propagated down from the constrained grid."""

    def test_battery_discharge_capped_by_distant_grid_limit(self):
        # The intermediate splitter (650 kW) is much larger than the 200 kW grid
        # connection, so only the grid limit should constrain net export.
        evse = _evse_with_kw(
            [100.0], voltage=400.0, max_current=750.0, shared_max_kw=750.0
        )
        battery = _battery(-500.0, max_kw=500.0)
        tree = _grid(_splitter(evse, battery, max_kw=650.0), max_kw=200.0)

        result = tree.distribute()
        split = result.connections[0]
        bat_out = split.connections[1]

        # 100 kW served locally + 200 kW exported => battery discharges 300 kW
        # (NOT its full 500 kW, which would push 400 kW through the 200 kW grid).
        assert float(bat_out.supplied_power) == pytest.approx(300.0)
        grid_net = float(result.requested_power - result.supplied_power)
        assert grid_net == pytest.approx(-200.0)
        _assert_no_exceeded(result)


class TestEVSESharedDischargeLimit:
    """Regression: charging scaled all ports together, but discharging applied the
    budget to each port independently, so an N-charger EVSE could discharge up to
    N times its shared rating."""

    def test_multi_charger_discharge_capped_to_shared_rating(self):
        # Two ports each driven to 300 kW discharge (600 kW raw) sharing one 300 kW
        # rating, with a generous export budget from above.
        evse = _evse_with_kw(
            [-300.0, -300.0],
            voltage=600.0,
            max_current=500.0,
            shared_max_kw=300.0,
        )
        out = evse.distribute(-1000.0)

        assert float(out.supplied_power) == pytest.approx(300.0)
        assert _kw(out) == pytest.approx([-150.0, -150.0])

    def test_multi_charger_discharge_capped_via_splitter(self):
        evse = _evse_with_kw(
            [-300.0, -300.0],
            voltage=600.0,
            max_current=500.0,
            shared_max_kw=300.0,
        )
        tree = _grid(_splitter(evse, max_kw=600.0), max_kw=1000.0)
        result = tree.distribute()
        evse_out = result.connections[0].connections[0]

        assert float(evse_out.supplied_power) <= 300.0 + 1e-4
        _assert_no_exceeded(result)


class TestPhantomSurplus:
    """Regression: parents estimated a child subtree's surplus/deficit from raw
    (uncapped) request/supply. An over-driven subtree whose charge and discharge
    both clamp to the same value has ~0 real net, but the raw numbers made it look
    like it had spare surplus, which a sibling load then consumed - overloading
    the grid."""

    def test_over_driven_subtree_creates_no_phantom_surplus(self):
        # Inner subtree: one EVSE over-driven charging (400 -> clamps to 300),
        # one over-driven discharging (600 -> clamps to 300). Real net ~= 0.
        charging = _evse_with_kw(
            [400.0], voltage=600.0, max_current=1000.0, shared_max_kw=300.0
        )
        discharging = _evse_with_kw(
            [-600.0], voltage=600.0, max_current=1000.0, shared_max_kw=300.0
        )
        inner = _splitter(charging, discharging, max_kw=600.0)
        # A sibling battery that would happily absorb any phantom surplus.
        battery = _battery(300.0, max_kw=500.0)
        tree = _grid(_splitter(inner, battery, max_kw=650.0), max_kw=200.0)

        result = tree.distribute()

        grid_net = float(result.requested_power - result.supplied_power)
        assert abs(grid_net) <= 200.0 + 1e-4
        _assert_no_exceeded(result)


class TestNetCapacityMetric:
    """Regression: exceeded capacity compared *gross* request and *gross* supply
    per node against its rating, so a battery discharging to feed a local sibling
    load (pure internal circulation) was wrongly counted as exceeding an upstream
    connection even though the *net* power crossing it was within limits."""

    def test_local_circulation_not_counted_as_exceeded(self):
        evse = _evse_with_kw([100.0], shared_max_kw=500.0)
        battery = _battery(-250.0, max_kw=500.0)
        # Splitter (300 kW) can source the 250 kW discharge; the grid connection
        # (200 kW) only sees the 150 kW net export.
        tree = _grid(_splitter(evse, battery, max_kw=300.0), max_kw=200.0)

        result = tree.distribute()
        split = result.connections[0]
        bat_out = split.connections[1]

        # Battery legitimately discharges the full 250 kW (100 local + 150 export).
        assert float(bat_out.supplied_power) == pytest.approx(250.0)
        # Net across the 200 kW grid connection is only 150 kW, even though the
        # gross discharge (250 kW) is above the grid rating -> not exceeded.
        grid_net = float(result.requested_power - result.supplied_power)
        assert grid_net == pytest.approx(-150.0)
        _assert_no_exceeded(result)


# Helpers for reading delivered power off a distributed tree ---------------------


def _requested(node) -> float:
    return float(node.requested_power)


def _throughput(node) -> float:
    """Signed throughput for a battery/passive (positive = draw, negative = supply)."""
    return float(node.throughput_now_kw)


class TestDeliveredPowerProportionalSharing:
    """When demand is oversubscribed, verify each EVSE/battery still receives its
    *proportional* share of the available capacity (not just that nothing exceeds).
    Grid is left effectively unconstrained so the inner splitter is the binding
    limit."""

    def test_two_evses_split_capacity_proportionally(self):
        # 300 + 100 = 400 kW demand through a 200 kW splitter -> scale 0.5.
        evse_a = _evse_with_kw([300.0], shared_max_kw=500.0)
        evse_b = _evse_with_kw([100.0], shared_max_kw=500.0)
        tree = _grid(_splitter(evse_a, evse_b, max_kw=200.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        assert _requested(split.connections[0]) == pytest.approx(150.0, abs=1e-2)
        assert _requested(split.connections[1]) == pytest.approx(50.0, abs=1e-2)
        # Conservation: total delivered equals the splitter capacity.
        assert _requested(split.connections[0]) + _requested(
            split.connections[1]
        ) == pytest.approx(200.0, abs=1e-2)

    def test_three_evses_keep_relative_ratios(self):
        # 100 : 200 : 300 through 300 kW -> scale 0.5 each.
        a = _evse_with_kw([100.0], shared_max_kw=500.0)
        b = _evse_with_kw([200.0], shared_max_kw=500.0)
        c = _evse_with_kw([300.0], shared_max_kw=500.0)
        tree = _grid(_splitter(a, b, c, max_kw=300.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        got = [_requested(ch) for ch in split.connections]
        assert got == pytest.approx([50.0, 100.0, 150.0], abs=1e-2)

    def test_evse_and_battery_charge_scaled_together(self):
        # EVSE 300 + battery charging 300 through 200 kW -> scale 1/3 each.
        evse = _evse_with_kw([300.0], shared_max_kw=500.0)
        battery = _battery(300.0)
        tree = _grid(_splitter(evse, battery, max_kw=200.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        assert _requested(split.connections[0]) == pytest.approx(100.0, abs=1e-2)
        assert _throughput(split.connections[1]) == pytest.approx(100.0, abs=1e-2)

    def test_undersubscribed_delivers_full_demand(self):
        # Everything fits -> nothing is scaled.
        evse = _evse_with_kw([120.0], shared_max_kw=500.0)
        battery = _battery(80.0)
        tree = _grid(_splitter(evse, battery, max_kw=500.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        assert _requested(split.connections[0]) == pytest.approx(120.0, abs=1e-2)
        assert _throughput(split.connections[1]) == pytest.approx(80.0, abs=1e-2)


class TestDeliveredPowerWithPassives:
    """Passive nodes are uncontrollable and take precedence. Verify controllable
    loads receive exactly the capacity left after passives, and that passive
    values themselves are never altered."""

    def test_passive_load_reduces_then_shares_remainder(self):
        # Passive load 50 -> 150 kW left for 300+100=400 demand -> scale 0.375.
        passive = _passive(50.0)
        evse_a = _evse_with_kw([300.0], shared_max_kw=500.0)
        evse_b = _evse_with_kw([100.0], shared_max_kw=500.0)
        tree = _grid(
            _splitter(passive, evse_a, evse_b, max_kw=200.0), max_kw=1_000_000.0
        )
        split = tree.distribute().connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(50.0, abs=1e-2)
        assert _requested(split.connections[1]) == pytest.approx(112.5, abs=1e-2)
        assert _requested(split.connections[2]) == pytest.approx(37.5, abs=1e-2)
        # Passive draw + controllable draw fills exactly the splitter capacity.
        total = (
            _throughput(split.connections[0])
            + _requested(split.connections[1])
            + _requested(split.connections[2])
        )
        assert total == pytest.approx(200.0, abs=1e-2)

    def test_passive_generation_frees_capacity_for_evse(self):
        # Passive generation of 100 kW lets a 150 kW EVSE charge fully even though
        # the cable only imports 50 kW net.
        passive = _passive(-100.0)
        evse = _evse_with_kw([150.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=200.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(-100.0, abs=1e-2)
        assert _requested(split.connections[1]) == pytest.approx(150.0, abs=1e-2)

    def test_passive_load_with_full_delivery_when_capacity_suffices(self):
        passive = _passive(30.0)
        evse = _evse_with_kw([100.0], shared_max_kw=500.0)
        battery = _battery(50.0)
        tree = _grid(
            _splitter(passive, evse, battery, max_kw=500.0), max_kw=1_000_000.0
        )
        split = tree.distribute().connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(30.0, abs=1e-2)
        assert _requested(split.connections[1]) == pytest.approx(100.0, abs=1e-2)
        assert _throughput(split.connections[2]) == pytest.approx(50.0, abs=1e-2)

    def test_passive_load_with_battery_discharge_delivers_expected(self):
        # Battery discharges 100 kW: 50 kW covers the passive load, 50 kW exports.
        passive = _passive(50.0)
        battery = _battery(-100.0)
        tree = _grid(_splitter(passive, battery, max_kw=300.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(50.0, abs=1e-2)
        assert _throughput(split.connections[1]) == pytest.approx(-100.0, abs=1e-2)


class TestDeliveredPowerWithDischargeSupport:
    """Verify charging that is (partly) served by local discharge still delivers
    the expected amount to the cars/batteries."""

    def test_charging_fully_met_using_battery_discharge(self):
        # EVSE wants 200 kW; 100 kW comes from the battery discharge, 100 kW from
        # the grid -> EVSE charges fully at 200 kW while net stays 100 kW.
        evse = _evse_with_kw([200.0], shared_max_kw=500.0)
        battery = _battery(-100.0)
        tree = _grid(_splitter(evse, battery, max_kw=200.0), max_kw=1_000_000.0)
        split = tree.distribute().connections[0]

        assert _requested(split.connections[0]) == pytest.approx(200.0, abs=1e-2)
        assert _throughput(split.connections[1]) == pytest.approx(-100.0, abs=1e-2)
        net = float(split.requested_power - split.supplied_power)
        assert net == pytest.approx(100.0, abs=1e-2)

    def test_internal_v2g_boosts_sibling_port_delivery(self):
        # One port discharges 100 kW, boosting the other port's charging so it can
        # reach 300 kW off a 200 kW allocation.
        evse = _evse_with_kw([400.0, -100.0], shared_max_kw=500.0)
        tree = _grid(_splitter(evse, max_kw=200.0), max_kw=1_000_000.0)
        evse_out = tree.distribute().connections[0].connections[0]

        assert _kw(evse_out) == pytest.approx([300.0, -100.0], abs=1e-2)


class TestPassiveForcedLoadsWithinCapacity:
    def test_single_passive_draw_with_evse(self):
        passive = _passive(80.0)
        evse = _evse_with_kw([100.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        _assert_no_exceeded(result, tol=1e-3)

    def test_single_passive_generation_with_evse(self):
        passive = _passive(-90.0)
        evse = _evse_with_kw([150.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        _assert_no_exceeded(result, tol=1e-3)

    def test_passive_with_evse_and_battery(self):
        passive = _passive(50.0)
        evse = _evse_with_kw([300.0], shared_max_kw=500.0)
        battery = _battery(-400.0)
        tree = _grid(_splitter(passive, evse, battery, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        _assert_no_exceeded(result, tol=1e-3)

    def test_multiple_passives_within_capacity(self):
        # A draw and a generation passive that partially offset each other.
        p_draw = _passive(90.0)
        p_gen = _passive(-30.0)
        evse = _evse_with_kw([120.0], shared_max_kw=500.0)
        tree = _grid(_splitter(p_draw, p_gen, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        _assert_no_exceeded(result, tol=1e-3)

    def test_passive_nested_in_multiple_levels(self):
        passive_outer = _passive(20.0)
        passive_inner = _passive(30.0)
        fast = _evse_with_kw(
            [200.0], voltage=600.0, max_current=500.0, shared_max_kw=600.0
        )
        battery = _battery(-100.0, max_kw=500.0)
        inner = _splitter(passive_inner, fast, battery, max_kw=150.0)
        tree = _grid(_splitter(passive_outer, inner, max_kw=300.0), max_kw=1000.0)
        result = tree.distribute()
        _assert_no_exceeded(result, tol=1e-3)

    def test_grid_connection_itself_bounded_by_passive_and_controllable(self):
        # The grid connection (not just the inner splitter) is the binding limit.
        passive = _passive(60.0)
        evse = _evse_with_kw([300.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=1000.0), max_kw=150.0)
        result = tree.distribute()
        assert _throughput(result.connections[0].connections[0]) == pytest.approx(
            60.0, abs=1e-2
        )
        assert _requested(result.connections[0].connections[1]) == pytest.approx(
            90.0, abs=1e-2
        )
        _assert_no_exceeded(result, tol=1e-3)

    def test_default_station_with_added_passive_load_runs_clean(self):
        # A realistic multi-level station (fast/slow chargers + battery) that
        # additionally carries a passive load (e.g. site facilities) well within
        # the branch's rating.
        passive = _passive(30.0)
        fast = _evse_with_kw(
            [200.0], voltage=600.0, max_current=500.0, shared_max_kw=600.0
        )
        slow = _evse_with_kw(
            [40.0], voltage=230.0, max_current=50.0, shared_max_kw=50.0
        )
        battery = _battery(-50.0, max_kw=500.0)
        tree = _grid(
            _splitter(passive, fast, slow, battery, max_kw=650.0), max_kw=200.0
        )
        result = tree.distribute()
        _assert_no_exceeded(result, tol=1e-3)


class TestPassiveForcedLoadsExceedStationCapacity:
    def test_passive_draw_alone_exceeds_splitter(self):
        passive = _passive(300.0)
        tree = _grid(_splitter(passive, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        # The passive is never throttled...
        assert _throughput(split.connections[0]) == pytest.approx(300.0)
        # ...so the shortfall at the splitter is exactly the physical overage.
        net = float(split.requested_power - split.supplied_power)
        assert net == pytest.approx(300.0)
        assert float(split.exceeded_power_all_children) == pytest.approx(
            100.0, abs=1e-3
        )

    def test_passive_generation_alone_exceeds_splitter(self):
        passive = _passive(-300.0)
        tree = _grid(_splitter(passive, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(-300.0)
        net = float(split.requested_power - split.supplied_power)
        assert net == pytest.approx(-300.0)
        assert float(split.exceeded_power_all_children) == pytest.approx(
            100.0, abs=1e-3
        )

    def test_passive_draw_exceeds_splitter_zeros_controllable_sibling(self):
        passive = _passive(300.0)
        evse = _evse_with_kw([100.0], shared_max_kw=500.0)
        tree = _grid(_splitter(passive, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        # Sibling demand cannot be served at all; it is driven to (approx) zero,
        # not left partially served or pushed negative.
        assert _requested(split.connections[1]) == pytest.approx(0.0, abs=1e-3)
        assert jnp.all(split.connections[1].charger_current_now >= -1e-6)
        # Exceedance equals exactly the passive overage; the EVSE being zeroed
        # does not add any further exceedance.
        assert float(split.exceeded_power_all_children) == pytest.approx(
            100.0, abs=1e-3
        )

    def test_passive_generation_exceeds_splitter_with_battery_unable_to_help(self):
        # Battery only wants to charge 50 kW; distribute() may scale that down
        # but never invents extra demand to soak up the passive surplus.
        passive = _passive(-300.0)
        battery = _battery(50.0)
        tree = _grid(_splitter(passive, battery, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(-300.0)
        assert _throughput(split.connections[1]) == pytest.approx(50.0, abs=1e-2)
        net = float(split.requested_power - split.supplied_power)
        assert net == pytest.approx(-250.0, abs=1e-2)
        assert float(split.exceeded_power_all_children) == pytest.approx(50.0, abs=1e-3)

    def test_multiple_passives_combined_exceed_splitter(self):
        # Individually each passive fits (120 < 200) but together they don't.
        p1 = _passive(120.0)
        p2 = _passive(120.0)
        evse = _evse_with_kw([50.0], shared_max_kw=500.0)
        tree = _grid(_splitter(p1, p2, evse, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(120.0)
        assert _throughput(split.connections[1]) == pytest.approx(120.0)
        assert _requested(split.connections[2]) == pytest.approx(0.0, abs=1e-3)
        assert float(split.exceeded_power_all_children) == pytest.approx(40.0, abs=1e-3)

    def test_passive_load_exceeds_distant_grid_connection(self):
        # The passive load sits deep in the tree; every splitter above it has
        # ample capacity, but the top-level grid connection cannot carry it.
        passive = _passive(400.0)
        inner = _splitter(passive, max_kw=1000.0)
        tree = _grid(_splitter(inner, max_kw=1000.0), max_kw=150.0)
        result = tree.distribute()

        assert _throughput(
            result.connections[0].connections[0].connections[0]
        ) == pytest.approx(400.0)
        grid_net = float(result.requested_power - result.supplied_power)
        assert grid_net == pytest.approx(400.0)
        assert float(result.exceeded_power_all_children) == pytest.approx(
            250.0, abs=1e-3
        )

    def test_passive_load_exceeds_grid_connection_directly(self):
        passive = _passive(500.0)
        tree = _grid(passive, max_kw=200.0)
        result = tree.distribute()

        assert _throughput(result.connections[0]) == pytest.approx(500.0)
        assert float(result.exceeded_power_all_children) == pytest.approx(
            300.0, abs=1e-3
        )


class TestPassiveFlexNodeCurtailment:
    def test_flex_generation_alone_curtailed_to_splitter_export_cap(self):
        flex = _passive_flex(-300.0)
        tree = _grid(_splitter(flex, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(-200.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)

    def test_flex_draw_alone_curtailed_to_splitter_import_cap(self):
        flex = _passive_flex(300.0)
        tree = _grid(_splitter(flex, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(200.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)

    def test_forced_draw_takes_precedence_over_flex_generation(self):
        # The shop (forced) always gets served in full; the PV (flex) is
        # curtailed to whatever export capacity remains.
        shop = _passive(50.0)
        pv = _passive_flex(-300.0)
        tree = _grid(_splitter(shop, pv, max_kw=200.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(50.0, abs=1e-2)
        assert _throughput(split.connections[1]) == pytest.approx(-200.0, abs=1e-2)
        net = float(split.requested_power - split.supplied_power)
        assert net == pytest.approx(-150.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)

    def test_flex_generation_shares_export_cap_proportionally_with_battery(self):
        # Flex PV and battery discharge each want 200 kW export through a
        # 200 kW splitter -> both scaled down to 100 kW (0.5 each).
        flex = _passive_flex(-200.0)
        battery = _battery(-200.0, max_kw=500.0)
        tree = _grid(_splitter(flex, battery, max_kw=200.0), max_kw=1_000_000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(-100.0, abs=1e-2)
        assert _throughput(split.connections[1]) == pytest.approx(-100.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)

    def test_flex_draw_shares_import_cap_proportionally_with_evse(self):
        flex = _passive_flex(300.0)
        evse = _evse_with_kw([100.0], shared_max_kw=500.0)
        tree = _grid(_splitter(flex, evse, max_kw=200.0), max_kw=1_000_000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(150.0, abs=1e-2)
        assert _requested(split.connections[1]) == pytest.approx(50.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)

    def test_flex_own_rating_binds_even_with_ample_splitter_capacity(self):
        # An inverter/curtailment rating of 100 kW on the flex node itself
        # limits it, even though the splitter above has plenty of headroom.
        flex = _passive_flex(-300.0, max_kw=100.0)
        tree = _grid(_splitter(flex, max_kw=1000.0), max_kw=1000.0)
        result = tree.distribute()
        split = result.connections[0]

        assert _throughput(split.connections[0]) == pytest.approx(-100.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)

    def test_flex_generation_exceeds_distant_grid_connection(self):
        # Even deep in the tree, a flex node is curtailed to respect a
        # constrained ancestor (the grid connection), not just its immediate
        # splitter.
        flex = _passive_flex(-400.0)
        inner = _splitter(flex, max_kw=1000.0)
        tree = _grid(_splitter(inner, max_kw=1000.0), max_kw=150.0)
        result = tree.distribute()

        assert _throughput(
            result.connections[0].connections[0].connections[0]
        ) == pytest.approx(-150.0, abs=1e-2)
        _assert_no_exceeded(result, tol=1e-3)


class TestDeliveredPowerNested:
    """Scaling caused by an outer splitter must still be shared proportionally
    among EVSEs living in different inner subtrees."""

    def test_outer_limit_shared_across_inner_subtrees(self):
        # Two inner subtrees each demand 200 kW; outer splitter caps at 300 kW ->
        # each inner EVSE receives 150 kW (0.75 scale).
        inner_a = _splitter(_evse_with_kw([200.0], shared_max_kw=500.0), max_kw=1000.0)
        inner_b = _splitter(_evse_with_kw([200.0], shared_max_kw=500.0), max_kw=1000.0)
        tree = _grid(_splitter(inner_a, inner_b, max_kw=300.0), max_kw=1_000_000.0)
        outer = tree.distribute().connections[0]

        a = _requested(outer.connections[0].connections[0])
        b = _requested(outer.connections[1].connections[0])
        assert a == pytest.approx(150.0, abs=1e-2)
        assert b == pytest.approx(150.0, abs=1e-2)
        assert a + b == pytest.approx(300.0, abs=1e-2)
