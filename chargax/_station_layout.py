from abc import abstractmethod
from dataclasses import fields
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


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


class StationBattery(StationNode):
    """
    A battery for the hub. Can be used to store excess energy or to provide energy to the grid.
    """

    capacity_kw: float = eqx.field(static=True)
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

    def distribute(self, available_from_top: float):
        total_available = available_from_top + self.supplied_power
        scale_factor = jnp.minimum(1.0, total_available / (self.requested_power + 1e-8))
        # Scale only charging (negative output); leave discharging untouched
        new_output = jnp.where(
            self.throughput_now_kw > 0,
            self.throughput_now_kw * scale_factor,
            self.throughput_now_kw,
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
        budget = jnp.maximum(available_from_top + self.supplied_power, 0.0)
        max_kw_throughput = self.max_kw_throughput[0]  # Max is shared
        budget = jnp.minimum(budget, max_kw_throughput)
        scale_factor = jnp.minimum(1.0, budget / (self.requested_power + 1e-8))

        # Scale only charging (positive) currents; leave discharging untouched
        new_current = jnp.where(
            self.charger_current_now > 0,
            self.charger_current_now * scale_factor,
            self.charger_current_now,
        )
        return self.replace(charger_current_now=new_current)


class StationSplitter(StationNode):
    """
    StationNode represents any combination of switchboards, cables, transformers, etc.
    A splitter can contain:
    - EVSEs
    - Batteries
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
    def exceeded_power_all_children(self) -> float:
        all_nodes = self._all_descendant_nodes
        requested = jnp.array([n.requested_power for n in all_nodes])
        supplied = jnp.array([n.supplied_power for n in all_nodes])
        max_kw = jnp.array(
            [
                n.max_kw_throughput[0] if isinstance(n, EVSE) else n.max_kw_throughput
                for n in all_nodes
            ]
        )
        return jnp.sum(jnp.maximum(requested - max_kw, supplied - max_kw))

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

        # Compute net flows as a single JAX array
        net_flows = jnp.array(
            [c.requested_power - c.supplied_power for c in self.connections]
        )

        available_power = jnp.minimum(available_from_top, self.max_kw_throughput)

        surplus = jnp.sum(jnp.maximum(0.0, -net_flows))  # energy supplied (V2G)
        deficit = jnp.sum(jnp.maximum(0.0, net_flows))  # energy demanded
        total_available = available_power + surplus
        scale_factor = jnp.minimum(1.0, total_available / (deficit + 1e-8))

        # Scale only demanding (positive) flows; leave supplying (negative) flows untouched
        scaled = jnp.where(net_flows > 0, net_flows * scale_factor, net_flows)

        return self.replace(
            connections=[c.distribute(net) for c, net in zip(self.connections, scaled)]
        )

    def update_evses_from_list(self, evses: List["EVSE"]) -> "StationSplitter":
        """Return a copy of this subtree with EVSEs replaced in order."""
        it = iter(evses)
        return jax.tree.map(
            lambda node: next(it) if isinstance(node, EVSE) else node,
            self,
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery)),
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
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery)),
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


class ChargingStation(StationSplitter):
    """The top-level charging station node (grid connection)"""

    def __post_init__(self):
        """Walk the tree once and pre-compute cumulative efficiency for each leaf node."""

        def _set_cumulative_efficiencies(node, parent_eff=1.0):
            eff = parent_eff * node.efficiency
            if isinstance(node, (EVSE, StationBattery)):
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
