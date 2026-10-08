from abc import abstractmethod
from dataclasses import fields
from typing import Any, Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from typing_extensions import Self


class StationNode(eqx.Module):
    max_kw_throughput: float
    efficiency: float

    def replace(self, **updates):
        keys, values = zip(*updates.items())
        return eqx.tree_at(lambda c: [c.__dict__[key] for key in keys], self, values)

    @property
    @abstractmethod
    def requested_power(self) -> float:
        """Returns the total power requested by this node/subtree in kW"""
        pass

    @property
    @abstractmethod
    def supplied_power(self) -> float:
        """Returns the total power supplied to the grid by this node/subtree in kW"""
        pass

    @abstractmethod
    def distribute(self, available_from_top: float):
        """Distribute the available power to the node/subtree"""
        pass


class _PassiveNode(StationNode):
    """Shared base for abstract sources / sinks. Provides or drains an uncontrollable amount of
    power from the system. This can be a constant amount or variable (e.g. based on time of day).

    Abstract base class. Use PassiveForcedNode or PassiveFlexNode to instantiate
    a passive node, depending on whether the flow must always take
    precedence or can be normalized alongside controllable loads by distribute().
    """

    load_profile: Callable[[Any], float] | float = eqx.field(static=True, default=0.0)
    throughput_now_kw: float = 0.0  # positive for draining, negative for supplying
    max_kw_throughput: float = eqx.field(default=jnp.iinfo(jnp.int32).max)
    efficiency: float = 1.0
    cumulative_efficiency: float = eqx.field(static=True, default=1.0)

    def get_current_load(self, state):
        if callable(self.load_profile):
            return self.load_profile(state)
        return self.load_profile

    @property
    def requested_power(self) -> float:
        """Power drawn from the grid in kW, always >= 0."""
        return jnp.maximum(0.0, self.throughput_now_kw)

    @property
    def supplied_power(self) -> float:
        """Power supplied back to the grid in kW, always >= 0."""
        return jnp.maximum(0.0, -self.throughput_now_kw)

    @abstractmethod
    def _feasible_net(self) -> float: ...

    @abstractmethod
    def distribute(self, available_from_top: float) -> Self: ...


class PassiveForcedNode(_PassiveNode):
    """An uncontrollable source / sink whose power is never normalized
    (scaled) by distribute() on a splitter - it takes precedence over
    controllable loads and is always drawn / supplied in full, even if that
    exceeds the station's capacity.

    Use this for must-serve loads (e.g. on-site shops that cannot have their
    power dropped due to EVs chargings, or must-run generation that cannot be curbed).
    """

    @property
    def _feasible_net(self) -> float:
        """Net power (requested - supplied) this node presents upstream. This
        flow is uncontrollable and always feasible from the node's own point
        of view; any shortfall shows up as exceeded capacity upstream instead."""
        return self.requested_power - self.supplied_power

    def distribute(self, available_from_top: float):
        return self  # never scaled - always takes precedence


class PassiveFlexNode(_PassiveNode):
    """An uncontrollable source / sink whose power *can* be curtailed by
    distribute() like any other controllable flow, when the station cannot
    carry it in full. Unlike PassiveForcedNode, this node's flow
    is scaled proportionally alongside EVSEs/batteries rather than reserved
    off the top.

    Use this for curtailable generation (e.g. PV that must be clipped when
    the grid connection or an inverter rating can't absorb it) or
    demand-response-capable loads.
    """

    @property
    def _feasible_net(self) -> float:
        """Net power this node can actually present upstream"""
        net = self.requested_power - self.supplied_power
        return jnp.clip(net, -self.max_kw_throughput, self.max_kw_throughput)

    def distribute(self, available_from_top: float):
        new_throughput = jnp.clip(
            available_from_top, -self.max_kw_throughput, self.max_kw_throughput
        )
        return self.replace(throughput_now_kw=new_throughput)


class StationBattery(StationNode):
    """
    A battery for the hub. Can be used to store excess energy or to provide energy to the grid.
    """

    capacity_kw: float
    throughput_now_kw: float = 0.0  # positive for charging, negative for discharging
    battery_now: float = 0.0
    tau: float = eqx.field(static=True, default=1.0)
    cumulative_efficiency: float = eqx.field(static=True, default=1.0)

    def __post_init__(self):
        # Start the battery at 25% charge
        object.__setattr__(self, "battery_now", self.capacity_kw * 0.25)

    @property
    def battery_percentage(self) -> float:
        return self.battery_now / self.capacity_kw

    @property
    def requested_power(self) -> float:
        """Power drawn from the grid (charging the battery) in kW, always >= 0."""
        return jnp.maximum(0.0, self.throughput_now_kw)

    @property
    def supplied_power(self) -> float:
        """Power supplied back to the grid (discharging the battery) in kW, always >= 0."""
        return jnp.maximum(0.0, -self.throughput_now_kw)

    @property
    def _feasible_net(self) -> float:
        """Net power this battery can actually present upstream, bounded by its
        own rating (it can neither charge nor discharge past max_kw_throughput)."""
        net = self.requested_power - self.supplied_power
        return jnp.clip(net, -self.max_kw_throughput, self.max_kw_throughput)

    def distribute(self, available_from_top: float):
        charging_budget = jnp.maximum(available_from_top, 0.0)
        discharge_budget = jnp.maximum(-available_from_top, 0.0)
        charge_scale = jnp.minimum(1.0, charging_budget / (self.requested_power + 1e-8))
        new_output = jnp.where(
            self.throughput_now_kw > 0,
            self.throughput_now_kw * charge_scale,
            jnp.where(
                available_from_top < 0,
                jnp.maximum(self.throughput_now_kw, -discharge_budget),
                self.throughput_now_kw,
            ),
        )
        new_output = jnp.clip(
            new_output, -self.max_kw_throughput, self.max_kw_throughput
        )
        return self.replace(throughput_now_kw=new_output)


class EVSE(StationNode):
    # Car variables
    car_time_till_leave: Array = eqx.field(converter=jnp.int_)
    car_battery_now_kw: Array
    car_battery_capacity_kw: Array
    car_desired_battery_percentage: Array
    car_arrival_battery_kw: Array  # To compensate / block the agent from discharging further than the arrival battery
    car_time_waited: Array
    charge_sensitive: Array = eqx.field(converter=jnp.bool_)  # False = Time sensitive

    # we need to keep track of the discharging per EV
    # as we discharge, and later charge agian, we can't have the
    # customer pay for the energy twice
    car_discharged_this_session_kw: Array

    car_ac_absolute_max_charge_rate_kw: Array
    car_ac_optimal_charge_threshold: Array
    car_dc_absolute_max_charge_rate_kw: Array
    car_dc_optimal_charge_threshold: Array

    # Charger variables
    charger_current_now: Array
    charger_is_car_connected: Array = eqx.field(converter=jnp.bool_)

    max_current: float  # = eqx.field(static=True)
    voltage: float  # = eqx.field(static=True)
    cumulative_efficiency: float  # = eqx.field(static=True, default=1.0)

    @property
    def is_dc(self) -> bool:  # Assumption: above 50 kW is DC
        return self.max_current * self.voltage / 1000.0 > 50.0

    @property
    def num_chargers(self) -> int:
        return self.car_battery_now_kw.size

    def __init__(
        self,
        num_chargers: int = 2,
        voltage: float = 230.0,
        max_current: float = 50.0,
        efficiency: float = 0.995,
    ):
        # Initialize all array fields to zeros
        for field in fields(self):
            setattr(self, field.name, jnp.zeros(num_chargers))

        self.max_kw_throughput = jnp.ones(num_chargers) * (
            (voltage * max_current) / 1000.0
        )  # max is shared, so distribute should scale by the number of chargers
        self.voltage = jnp.ones(num_chargers) * voltage
        self.max_current = jnp.ones(num_chargers) * max_current
        self.efficiency = jnp.ones(num_chargers) * efficiency
        self.cumulative_efficiency = 1.0  # Set by ChargingStation.__post_init__

    @property
    def car_battery_percentage(self) -> Array:
        return self.car_battery_now_kw / (self.car_battery_capacity_kw + 1e-8)

    @property
    def car_battery_desired_remaining(self) -> Array:
        return self.car_desired_battery_percentage - self.car_battery_percentage

    @property
    def car_battery_desired_remaining_kw(self) -> Array:
        desired_battery_kw = (
            self.car_desired_battery_percentage * self.car_battery_capacity_kw
        )
        return desired_battery_kw - self.car_battery_now_kw

    @property
    def power_output(self) -> Array:
        """Returns the power output in kW, positive for charging, negative for discharging"""
        return (self.voltage * self.charger_current_now) / 1000.0

    @property
    def requested_power(self):
        """Returns the requested power in kW"""
        return jnp.sum(jnp.maximum(0.0, self.power_output))

    @property
    def supplied_power(self):
        """Returns the supplied power in kW (V2G), always >= 0."""
        return jnp.sum(jnp.maximum(0.0, -self.power_output))

    @property
    def _feasible_net(self):
        """Net power this EVSE can actually present upstream. Charge/discharge on
        different ports of the same EVSE offset each other internally.
        The rating is shared across chargers."""
        max_kw = self.max_kw_throughput[0]
        net = self.requested_power - self.supplied_power
        return jnp.clip(net, -max_kw, max_kw)

    @property
    def car_max_current_intake(self) -> Array:
        return self._car_max_current(self.car_battery_percentage)

    @property
    def car_max_current_outtake(self) -> Array:
        return self._car_max_current(1 - self.car_battery_percentage)

    def _car_max_current(self, battery_percentage: Array) -> Array:
        tau, abs_max_rate = jax.tree.map(
            lambda x, y: jnp.where(self.is_dc, x, y),
            (
                self.car_dc_optimal_charge_threshold,
                self.car_dc_absolute_max_charge_rate_kw,
            ),
            (
                self.car_ac_optimal_charge_threshold,
                self.car_ac_absolute_max_charge_rate_kw,
            ),
        )
        # linearly decay the charge rate to 5% after reaching the threshold
        max_charge_rate_kw = (
            jnp.where(
                battery_percentage > tau,
                abs_max_rate * (1 - (battery_percentage - tau) / (1 - tau) + 0.10),
                abs_max_rate,
            )
            * self.charger_is_car_connected
        )  # charge rate is 0 if car is not connected
        max_charge_rate_w = max_charge_rate_kw * 1000.0
        return max_charge_rate_w / (
            self.voltage + 1e-8
        )  # add small value to avoid division by zero

    def distribute(self, available_from_top: float):
        max_kw = self.max_kw_throughput[0]  # shared across chargers
        # Charging side: charging may be boosted by simultaneous discharge on
        # sibling ports (internal V2G) but never exceeds the shared rating.
        charging_budget = jnp.minimum(
            jnp.maximum(available_from_top + self.supplied_power, 0.0), max_kw
        )
        charge_scale = jnp.minimum(1.0, charging_budget / (self.requested_power + 1e-8))
        charge_met = self.requested_power * charge_scale
        export_budget = jnp.maximum(-available_from_top, 0.0)
        max_supply = jnp.minimum(max_kw, charge_met + export_budget)
        discharge_scale = jnp.minimum(1.0, max_supply / (self.supplied_power + 1e-8))

        new_current = jnp.where(
            self.charger_current_now > 0,
            self.charger_current_now * charge_scale,
            self.charger_current_now * discharge_scale,
        )
        return self.replace(charger_current_now=new_current)


class StationSplitter(StationNode):
    """
    StationNode represents any combination of switchboards, cables, transformers, etc.
    A splitter can contain:
    - EVSEs
    - Batteries
    - PassiveForcedNode / PassiveFlexNode instances
    - Other nodes
    """

    connections: List[StationNode]

    @property
    def evses(self) -> List[EVSE]:
        """Return a list of all EVSEs in this subtree."""
        return [
            evse
            for evse in jax.tree.leaves(
                self.connections, is_leaf=lambda x: isinstance(x, EVSE)
            )
            if isinstance(evse, EVSE)
        ]

    @property
    def evses_flat(self) -> EVSE:
        """Return a single EVSE object with all chargers concatenated. The order of chargers is the same as in evses."""
        evses = jax.tree.map(jnp.atleast_1d, self.evses)  # for concatenation
        return jax.tree.map(lambda *t: jnp.concatenate(t), *evses)

    @property
    def batteries(self) -> List["StationBattery"]:
        """Return a list of all batteries in this subtree."""
        return [
            battery
            for battery in jax.tree.leaves(
                self.connections, is_leaf=lambda x: isinstance(x, StationBattery)
            )
            if isinstance(battery, StationBattery)
        ]

    @property
    def batteries_flat(self) -> "StationBattery":
        """Return a single StationBattery object with all batteries concatenated. The order of batteries is the same as in batteries."""
        if not self.batteries:
            return StationBattery(0, 0, 0)  # dummy battery for compatibility
        batteries = jax.tree.map(jnp.atleast_1d, self.batteries)
        return jax.tree.map(lambda *t: jnp.concatenate(t), *batteries)

    @property
    def passives(self) -> List["_PassiveNode"]:
        """Return a list of all passive nodes (forced and flex) in this subtree."""
        return [
            passive
            for passive in jax.tree.leaves(
                self.connections, is_leaf=lambda x: isinstance(x, _PassiveNode)
            )
            if isinstance(passive, _PassiveNode)
        ]

    @property
    def passives_flat(self) -> "_PassiveNode":
        """Return a single passive node with all passive loads concatenated.

        All passives in this subtree must be the same concrete type
        (all ``PassiveForcedNode`` or all ``PassiveFlexNode``), since
        concatenation requires matching pytree structure.

        Raises:
            ValueError: If the subtree contains a mix of passive node types.
                Use :attr:`passives` instead and handle each node separately.
        """
        passives = self.passives
        if not passives:
            return PassiveForcedNode(load_profile=0.0)

        passive_types = {type(p) for p in passives}
        if len(passive_types) > 1:
            type_names = ", ".join(sorted(t.__name__ for t in passive_types))
            raise ValueError(
                "passives_flat() cannot concatenate mixed passive node types "
                f"({type_names}). Use the passives property instead and handle "
                "each node separately."
            )

        passives = jax.tree.map(jnp.atleast_1d, passives)
        return jax.tree.map(lambda *t: jnp.concatenate(t), *passives)

    @property
    def num_chargers(self) -> int:
        """Return the total number of chargers in this subtree."""
        return sum(evse.num_chargers for evse in self.evses)

    @property
    def _all_descendant_nodes(self) -> List[StationNode]:
        """Recursively collect all StationNodes below this node (children, grandchildren, etc.)."""
        result = []
        for c in self.connections:
            result.append(c)
            if isinstance(c, StationSplitter):
                result.extend(c._all_descendant_nodes)
        return result

    @property
    def requested_power(self) -> float:
        return sum(c.requested_power for c in self.connections)

    @property
    def supplied_power(self) -> float:
        return sum(c.supplied_power for c in self.connections)

    @property
    def _feasible_net(self) -> float:
        """Net power this subtree can actually present to its parent.

        At most ``min(deficit, max_kw)`` of demand can be met and at most
        ``min(surplus, max_kw)`` of supply can be sourced, so the leftover that
        actually crosses to the parent connection is:

            net = max(0, min(deficit, cap) - surplus)      # residual import
                - max(0, min(surplus, cap) - deficit)      # residual export
        """
        nets = jnp.array([c._feasible_net for c in self.connections])
        deficit = jnp.sum(jnp.maximum(0.0, nets))
        surplus = jnp.sum(jnp.maximum(0.0, -nets))
        cap = self.max_kw_throughput
        residual_import = jnp.maximum(0.0, jnp.minimum(deficit, cap) - surplus)
        residual_export = jnp.maximum(0.0, jnp.minimum(surplus, cap) - deficit)
        return residual_import - residual_export

    @property
    def exceeded_power_all_children(self) -> float:
        all_nodes = [self] + self._all_descendant_nodes
        net = jnp.array([n.requested_power - n.supplied_power for n in all_nodes])
        max_kw = jnp.array(
            [
                n.max_kw_throughput[0] if isinstance(n, EVSE) else n.max_kw_throughput
                for n in all_nodes
            ]
        )
        return jnp.sum(jnp.maximum(0.0, jnp.abs(net) - max_kw))

    def cumulative_efficiency_of(
        self, target: "EVSE | StationBattery", parent_efficiency: float = 1.0
    ) -> float:
        """Find the cumulative efficiency from root to a specific leaf node."""
        efficiency = parent_efficiency * self.efficiency
        for c in self.connections:
            if c is target:
                return efficiency * c.efficiency
            if isinstance(c, StationSplitter):
                result = c.cumulative_efficiency_of(target, efficiency)
                if result is not None:
                    return result
        return None

    def distribute(self, available_from_top: float | None = None):

        if available_from_top is None:  # Called on grid connection
            available_from_top = self.max_kw_throughput
            # The grid connection itself may export up to its own rating.
            export_budget = self.max_kw_throughput
        else:
            export_budget = jnp.maximum(0.0, -available_from_top)

        connections = self.connections
        # Only PassiveForcedNode is reserved off the top / never scaled.
        # PassiveFlexNode flows through the same controllable pathway as
        # EVSEs/batteries below, and can be curtailed like them.
        passive_mask = jnp.array(
            [isinstance(c, PassiveForcedNode) for c in connections]
        )
        net_flows = jnp.array([c._feasible_net for c in connections])

        passive_requested = sum(
            c.requested_power for c in connections if isinstance(c, PassiveForcedNode)
        )
        passive_supplied = sum(
            c.supplied_power for c in connections if isinstance(c, PassiveForcedNode)
        )

        available_power = jnp.minimum(available_from_top, self.max_kw_throughput)
        # Reserve passive flows first; only scale controllable children
        available_after_passive = available_power - passive_requested + passive_supplied

        controllable_net = jnp.where(passive_mask, 0.0, net_flows)
        surplus = jnp.sum(jnp.maximum(0.0, -controllable_net))
        deficit = jnp.sum(jnp.maximum(0.0, controllable_net))
        total_available = jnp.minimum(
            available_after_passive + surplus, self.max_kw_throughput
        )
        scale_factor = jnp.minimum(1.0, total_available / (deficit + 1e-8))

        # Pass 1: scale demanding controllable flows; passive nets stay fixed
        scaled = jnp.where(
            passive_mask,
            net_flows,
            jnp.where(net_flows > 0, net_flows * scale_factor, net_flows),
        )

        # Pass 2: cap controllable supply only (passive export is never scaled)
        scaled_deficit = jnp.sum(jnp.maximum(0.0, scaled))
        controllable_supply_total = jnp.sum(jnp.maximum(0.0, -controllable_net))

        # (a) The busbar can source at most its own rating (local circulation).
        gross_supply_cap = scaled_deficit + jnp.maximum(
            0.0, self.max_kw_throughput - scaled_deficit
        )
        # (b) The net power pushed upstream (requested - supplied) must not drop
        #     below -export_budget, otherwise the parent connection is overloaded.
        #     net = passive_net + demand_met - controllable_supply >= -export_budget
        passive_net = jnp.sum(jnp.where(passive_mask, net_flows, 0.0))
        demand_met = jnp.sum(jnp.where((~passive_mask) & (net_flows > 0), scaled, 0.0))
        net_export_cap = jnp.maximum(0.0, passive_net + demand_met + export_budget)

        max_supply_allowed = jnp.minimum(gross_supply_cap, net_export_cap)
        supply_scale = jnp.minimum(
            1.0, max_supply_allowed / (controllable_supply_total + 1e-8)
        )
        scaled = jnp.where(
            passive_mask | (net_flows >= 0),
            scaled,
            scaled * supply_scale,
        )

        return self.replace(
            connections=[c.distribute(net) for c, net in zip(connections, scaled)]
        )

    def update_evses_from_list(self, evses: List["EVSE"]) -> "StationSplitter":
        """Return a copy of this subtree with EVSEs replaced in order."""
        it = iter(evses)
        return jax.tree.map(
            lambda node: next(it) if isinstance(node, EVSE) else node,
            self,
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery, _PassiveNode)),
        )

    def update_evses_from_flat(self, flat_evse: EVSE) -> "StationSplitter":
        """Split a flat EVSE back into per-EVSE nodes and put them in the tree."""
        sizes = tuple(e.num_chargers for e in self.evses)
        split_indices = np.cumsum(sizes[:-1])

        # Flatten to raw leaf arrays + structure
        leaves, treedef = jax.tree.flatten(flat_evse)
        split_leaves = [jnp.split(leaf, split_indices) for leaf in leaves]

        # Transpose: for each EVSE index, gather its slice of every leaf and unflatten
        evses = [
            jax.tree.unflatten(
                treedef, [split_leaves[j][i] for j in range(len(leaves))]
            )
            for i in range(len(sizes))
        ]
        return self.update_evses_from_list(evses)

    def update_batteries_from_list(
        self, batteries: List["StationBattery"]
    ) -> "StationSplitter":
        """Return a copy of this subtree with Batteries replaced in order."""
        it = iter(batteries)
        return jax.tree.map(
            lambda node: next(it) if isinstance(node, StationBattery) else node,
            self,
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery, _PassiveNode)),
        )

    def update_batteries_from_flat(
        self, flat_battery: "StationBattery"
    ) -> "StationSplitter":
        """Split a flat StationBattery back into per-battery nodes and put them in the tree."""
        if not self.batteries:
            return self  # no batteries to update, return original tree

        n = len(self.batteries)

        # Flatten to raw leaf arrays + structure
        leaves, treedef = jax.tree.flatten(flat_battery)
        split_indices = np.arange(1, n)
        split_leaves = [jnp.split(leaf, split_indices) for leaf in leaves]

        # Transpose: for each Battery index, gather its slice of every leaf,
        # squeeze back to scalar (batteries have scalar fields), and unflatten
        batteries = [
            jax.tree.unflatten(
                treedef, [split_leaves[j][i].squeeze() for j in range(len(leaves))]
            )
            for i in range(n)
        ]
        return self.update_batteries_from_list(batteries)

    def update_passives_from_list(
        self, passives: List["_PassiveNode"]
    ) -> "StationSplitter":
        """Return a copy of this subtree with passive nodes replaced in order."""
        it = iter(passives)
        return jax.tree.map(
            lambda node: next(it) if isinstance(node, _PassiveNode) else node,
            self,
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery, _PassiveNode)),
        )


class ChargingStation(StationSplitter):
    """The top-level charging station node (grid connection)"""

    def __post_init__(self):
        """Walk the tree once and pre-compute cumulative efficiency for each leaf node."""

        def _set_cumulative_efficiencies(node, parent_eff=1.0):
            eff = parent_eff * node.efficiency
            if isinstance(node, (EVSE, StationBattery, _PassiveNode)):
                object.__setattr__(node, "cumulative_efficiency", eff)
                return
            if isinstance(node, StationSplitter):
                for c in node.connections:
                    _set_cumulative_efficiencies(c, eff)

        _set_cumulative_efficiencies(self)

    @classmethod
    def init_default_station(cls) -> "ChargingStation":
        """Initializes a station layout with a mix of fast and slow chargers and a battery on site.
        This site has a constrained grid connection and thus requires the battery to meet demand during peak hours.
        """
        return cls(
            max_kw_throughput=200.0,  # Grid connection max throughput
            efficiency=1.0,
            connections=[
                StationSplitter(
                    max_kw_throughput=650.0,
                    efficiency=0.995,
                    connections=[
                        # Fast charger:
                        StationSplitter(
                            max_kw_throughput=600.0,
                            efficiency=0.995,
                            connections=[
                                EVSE(
                                    voltage=600,
                                    max_current=500,
                                    num_chargers=2,
                                    efficiency=0.995,
                                ),
                                EVSE(
                                    voltage=600,
                                    max_current=500,
                                    num_chargers=2,
                                    efficiency=0.995,
                                ),
                            ],
                        ),
                        # Slow charger:
                        StationSplitter(
                            max_kw_throughput=50.0,
                            efficiency=0.995,
                            connections=[
                                EVSE(
                                    voltage=230,
                                    max_current=50,
                                    num_chargers=2,
                                    efficiency=0.995,
                                )
                            ],
                        ),
                        # Battery on site:
                        StationBattery(
                            capacity_kw=2500.0,
                            max_kw_throughput=500.0,
                            efficiency=0.995,
                        ),
                    ],
                ),
            ],
        )
