import chex
from typing import Union, List, Optional, Literal
import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx
from dataclasses import replace, fields
from ._data_loaders import get_car_data

@chex.dataclass(frozen=True)
class CarProfiles:
    frequencies: chex.Array
    threshold_tau: chex.Array
    capacity: chex.Array
    ac_max_rate: chex.Array
    dc_max_rate: chex.Array

class ChargersState(eqx.Module):
    # Car variables
    car_time_till_leave: chex.Array
    car_battery_now: chex.Array
    car_battery_capacity: chex.Array
    car_desired_battery_percentage: chex.Array
    charge_sensitive: chex.Array # False = Time sensitive

    car_ac_absolute_max_charge_rate_kw: chex.Array
    car_ac_optimal_charge_threshold: chex.Array
    car_dc_absolute_max_charge_rate_kw: chex.Array
    car_dc_optimal_charge_threshold: chex.Array
    
    # Charger variables
    charger_current_now: chex.Array
    charger_is_car_connected: chex.Array
    charger_voltage: np.ndarray # We assume this is constant for all chargers
    charger_is_dc: np.ndarray 
    # max current is set in the EVSE

    @property
    def car_battery_percentage(self) -> jnp.ndarray:
        return self.car_battery_now / (self.car_battery_capacity + 1e-8)
    
    @property
    def car_battery_desired_remaining(self) -> jnp.ndarray:
        return jnp.maximum(0, self.car_desired_battery_percentage - self.car_battery_percentage)
    
    @property
    def charger_output_now_kw(self) -> jnp.ndarray:
        return (self.charger_voltage * self.charger_current_now) / 1000.0
    
    @property
    def charger_throughput_now_kw(self) -> jnp.ndarray:
        """ Uses absolute value of current to calculate throughput """
        return (self.charger_voltage * jnp.abs(self.charger_current_now)) / 1000.0
    
    @property
    def car_max_current_intake(self) -> jnp.ndarray:
        tau, abs_max_rate = jax.tree.map(
            lambda x, y: jnp.where(
                self.charger_is_dc, x, y
            ), 
            (self.car_dc_optimal_charge_threshold, self.car_dc_absolute_max_charge_rate_kw),
            (self.car_ac_optimal_charge_threshold, self.car_ac_absolute_max_charge_rate_kw)
        )
        # linearly decay the charge rate to 5% after reaching the threshold
        max_charge_rate_kw = jnp.where(
            self.car_battery_percentage > tau,
            abs_max_rate * (1 - (self.car_battery_percentage - tau) / (1 - tau) + 0.10),
            abs_max_rate
        ) * self.charger_is_car_connected # charge rate is 0 if car is not connected
        max_charge_rate_w = max_charge_rate_kw * 1000.0
        return (max_charge_rate_w / (self.charger_voltage + 1e-8)) # add small value to avoid division by zero

    def __init__(
            self, 
            station: 'ChargingStation' = None, 
            sample_method: Literal["empty", "eu", "us", "world", "custom"] = "empty",
            key: Optional[chex.PRNGKey] = None,
            **kwargs
        ):
        if kwargs: # To allow for replace(..., **kwargs)
            self.__dict__.update(kwargs)
            return
        
        num_chargers = station.num_chargers

        if sample_method == "empty":
            for field in fields(self):
                setattr(self, field.name, jnp.zeros(num_chargers))
        
        elif sample_method == "random":
            assert key is not None, "Key must be provided for random sampling"
            keys = jax.random.split(key, 8)
            self.car_time_till_leave = jax.random.randint(keys[0], (num_chargers,), 80, 140)
            self.car_battery_now = jax.random.uniform(keys[1], (num_chargers,), minval=0.10, maxval=0.60)
            self.car_ac_absolute_max_charge_rate_kw = jax.random.uniform(keys[2], (num_chargers,), minval=100., maxval=100.)
            self.car_ac_optimal_charge_threshold = jax.random.uniform(keys[3], (num_chargers,), minval=0.8, maxval=0.8)
            self.car_dc_absolute_max_charge_rate_kw = jax.random.uniform(keys[4], (num_chargers,), minval=100., maxval=100.)
            self.car_dc_optimal_charge_threshold = jax.random.uniform(keys[5], (num_chargers,), minval=0.8, maxval=0.8)
            self.car_battery_capacity = jax.random.uniform(keys[6], (num_chargers,), minval=200., maxval=299.)
            self.car_desired_battery_percentage = jax.random.uniform(keys[6], (num_chargers,), minval=0.8, maxval=0.9)
            self.charge_sensitive = jax.random.randint(keys[7], (num_chargers,), 0, 1).astype(bool)
        else:
            car_data = sample_method
        # elif sample_method in ["eu", "us", "world"]:
            # car_data = jnp.array(get_car_data(sample_method))
            probs = car_data[:, 0]
            cars = jax.random.choice(key, probs.size, shape=(num_chargers,), replace=True, p=probs)
            tau_car_data = car_data[:, 1]
            capacity_car_data = car_data[:, 2]
            ac_max_rate_car_data = car_data[:, 3]
            dc_max_rate_car_data = car_data[:, 4]
            self.car_ac_absolute_max_charge_rate_kw = ac_max_rate_car_data[cars]
            self.car_ac_optimal_charge_threshold = tau_car_data[cars]
            self.car_dc_absolute_max_charge_rate_kw = dc_max_rate_car_data[cars]
            self.car_dc_optimal_charge_threshold = tau_car_data[cars]
            self.car_battery_capacity = capacity_car_data[cars]

            keys = jax.random.split(key, 8)
            self.car_time_till_leave = jax.random.randint(keys[0], (num_chargers,), 80, 140)
            self.car_battery_now = jax.random.uniform(keys[1], (num_chargers,), minval=0.10, maxval=0.60)
            self.car_desired_battery_percentage = jax.random.uniform(keys[6], (num_chargers,), minval=0.8, maxval=0.9)
            self.charge_sensitive = jax.random.randint(keys[7], (num_chargers,), 0, 1)
        
        # set types
        self.car_time_till_leave = self.car_time_till_leave.astype(int)
        self.charge_sensitive = self.charge_sensitive.astype(bool)
        
        # set charger voltage and dc/ac
        rated_voltages = [np.repeat(evse.voltage_rated, len(evse.connections)) for evse in station.evses]
        max_kw = [np.repeat(evse.group_capacity_max_kw, len(evse.connections)) for evse in station.evses]
        voltages_per_charger = np.concatenate(rated_voltages)
        max_kw_per_charger = np.concatenate(max_kw)
        self.charger_voltage = voltages_per_charger
        self.charger_is_dc = max_kw_per_charger > 50.0

        # init these empty for every sample method
        self.charger_current_now = jnp.zeros(num_chargers)
        self.charger_is_car_connected = jnp.zeros(num_chargers, dtype=bool)
        

class StationSplitter(eqx.Module):
    """
    A splitter represents any combination of switchboards, cables, transformers,
    or other auxiliary equipment that may induce a loss and is part of the network of
    chargers.
    A splitter can contain:
    - EVSEs
    - Other Splitters
    """
    connections: List[Union['StationEVSE', 'StationSplitter']]
    group_capacity_max_kw: float = eqx.field(static=True)
    efficiency: float = eqx.field(static=True, default=0.995)

    @property
    def charger_ids_per_children_evse(self) -> np.ndarray:
        return jax.tree.leaves(self.connections)
    
    @property
    def charger_ids_children(self) -> np.ndarray:
        return np.concatenate(self.charger_ids_per_children_evse)
    
    @property
    def number_of_chargers_children(self) -> int:
        return len(self.charger_ids_children)
    
    @property
    def evses_children(self) -> List['StationEVSE']:
        return jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, StationEVSE))
    
    @property
    def splitters_children(self) -> List['StationSplitter']:
        return jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, StationSplitter))

    # @property
    # def evse_per_chargepoint_children(self) -> List['StationEVSE']:
    #     """ 
    #         Returns a list of EVSEs of length num_chargers (that are children of this group).
    #         Each index of the list corresponds to the charger index and contains the 
    #         EVSE that is the parent connection of the charger.
    #     """
    #     EVSEs = self.evse_children
    #     evse_per_chargepoint = []
    #     for evse in EVSEs:
    #         evse_per_chargepoint.extend([evse] * len(evse.connections))
    #     return evse_per_chargepoint

    @property
    def efficiency_per_charger(self) -> jnp.ndarray:
        efficiency_to_chargepoint = np.ones(self.number_of_chargers_children)
        for path, charger_ids in jax.tree.leaves_with_path(self):
            curr_node = self
            for node in path[:-1]: # omit the last node which is the charger
                if isinstance(node, jax.tree_util.GetAttrKey):
                    curr_node = curr_node.connections
                elif isinstance(node, jax.tree_util.SequenceKey):
                    curr_node = curr_node[node.idx]
                    efficiency_to_chargepoint[charger_ids] *= (
                        curr_node.efficiency
                    )
                    
        efficiency_to_chargepoint *= self.efficiency

        return efficiency_to_chargepoint
    
    # def total_draw_or_supply_kw(self, charger_state: 'ChargersState') -> float:
    #     children_charger_outputs = charger_state.charger_output_now_kw[self.charger_ids_children]
    #     return jnp.sum(children_charger_outputs)
    
    # def total_kw_to_customers(self, charger_state: 'ChargersState') -> float:
    #     children_charger_outputs = charger_state.charger_output_now_kw[self.charger_ids_children]
    #     charger_outputs_ex_discharge = jnp.maximum(children_charger_outputs, 0)
    #     return jnp.sum(charger_outputs_ex_discharge)
    
    # def total_kw_to_grid(self, charger_state: 'ChargersState') -> float:
    #     children_charger_outputs = charger_state.charger_output_now_kw[self.charger_ids_children]
    #     charger_outputs_ex_charge = jnp.minimum(children_charger_outputs, 0)
    #     return jnp.sum(jnp.abs(charger_outputs_ex_charge))

    # def total_power_output_kw(self, charger_state: 'ChargersState') -> float:
    #     children_charger_outputs = charger_state.charger_output_now_kw[self.charger_ids_children]
    #     return jnp.sum(children_charger_outputs)
    
    def total_kw_throughput(self, charger_state: 'ChargersState') -> float:
        children_charger_outputs = charger_state.charger_throughput_now_kw[self.charger_ids_children]
        return jnp.sum(children_charger_outputs)
    
    # def total_power_loss_kw(self, charger_state: 'ChargersState') -> float:
    #     losses = self.total_power_throughput_kw(charger_state) / self.efficiency_per_charger
    #     return jnp.sum(losses)
    
    # def total_power_draw_kw(self, charger_state: 'ChargersState') -> float:
    #     return self.total_power_output_kw(charger_state) + self.total_power_loss_kw(charger_state)    
    
    def normalize_currents(self, charger_state: 'ChargersState') -> 'ChargersState':

        max_capacity = self.group_capacity_max_kw
        curr_load = self.total_kw_throughput(charger_state)
        normalization_factor = jax.lax.select(
            curr_load > max_capacity,
            max_capacity / curr_load,
            1.
        )
        currents = charger_state.charger_current_now
        currents = currents.at[self.charger_ids_children].set(
            currents[self.charger_ids_children] * normalization_factor
        )
        return replace(
            charger_state,
            charger_current_now=currents
        )
    
    def get_parent(self, root: 'StationSplitter'):
        """
        Find the parent of the current StationSplitter instance starting from the root.

        Args:
            root (StationSplitter): The root of the tree to start the search from.
        
        Returns:
            Optional[StationSplitter]: The parent of the current instance if found, otherwise None.
        """
        for connection in root.connections:
            if isinstance(connection, StationSplitter):
                if self in connection.connections:
                    return connection
                parent = self.get_parent(connection)
                if parent:
                    return parent
        return None
    
class StationEVSE(StationSplitter):
    """
    An EVSE -- the final splitter in the hierarchy -- is a splitter that connects to the chargers.
    """
    connections: np.ndarray = eqx.field(converter=np.asarray) # Charger indices
    voltage_rated: float = eqx.field(static=True)
    current_max: float = eqx.field(static=True)

    def __init__(self, connections: np.ndarray, voltage_rated: float = 230.0, current_max: float = 50.0, **kwargs):
        self.__dict__.update(kwargs)
        self.connections = connections
        self.voltage_rated = voltage_rated
        self.current_max = current_max
        self.group_capacity_max_kw = (self.voltage_rated * self.current_max) / 1000.0

class StationBattery(eqx.Module):
    """
    A battery for the hub. Can be used to store excess energy or to provide energy to the grid.
    """
    capacity_kw: float = 100000.0
    battery_now: float = 200.0
    max_rate_kw: float = 500.0

    @property
    def battery_percentage(self) -> float:
        return self.battery_now / self.capacity_kw

class ChargingStation(eqx.Module):
    """
        The hub is the top level of the charger hierarchy.
        It contains the topology of the chargers as a tree of ChargerNodes,
        where the root node is the grid connection.
        As well as a single object containing the state of all chargers.
    """
    charger_layout: StationSplitter

    def __init__(self, num_chargers: int = 16, num_chargers_per_group: int = 2, num_dc_groups: int = 4):
        assert num_chargers % num_chargers_per_group == 0, "Chargers must be divisible by chargers_per_group"
        assert num_chargers_per_group >= 1, "Chargers per group must be greater than 0"
        assert num_chargers > num_chargers_per_group, "Chargers must be greater than chargers_per_group"

        charger_indices = np.arange(num_chargers)
        charger_indices = charger_indices.reshape(-1, num_chargers_per_group)

        DC_EVSEs = [
            StationEVSE(connections=ci, voltage_rated=500.0, current_max=300.0) for ci in charger_indices[:num_dc_groups]
        ]
        AC_EVSEs = [
            StationEVSE(connections=ci) for ci in charger_indices[num_dc_groups:] # default is set to 230V, 50A (11.5kW)
        ]
        EVSEs = DC_EVSEs + AC_EVSEs
        
        combined_total_capacity = sum([group.group_capacity_max_kw for group in EVSEs])
        grid_connection_node = StationSplitter(
            connections=EVSEs,
            group_capacity_max_kw=combined_total_capacity
        )

        self.charger_layout = grid_connection_node

    @property
    def root(self) -> StationSplitter:
        """ Convenience method to get the root node of the charger layout """
        return self.charger_layout

    @property
    def num_chargers(self) -> int:
        return self.charger_layout.number_of_chargers_children
    
    @property
    def charger_ids(self) -> np.ndarray:
        return self.charger_layout.charger_ids_children
    
    @property
    def charger_ids_per_evse(self) -> np.ndarray:
        return self.charger_layout.charger_ids_per_children_evse
    
    @property
    def evses(self) -> List[StationEVSE]:
        return self.charger_layout.evses_children
    
    @property
    def splitters(self) -> List[StationSplitter]:
        return self.charger_layout.splitters_children
    
    # @property
    # def evse_per_chargepoint(self) -> List['StationEVSE']:
    #     return self.charger_layout.evse_per_chargepoint_children

@chex.dataclass(frozen=True)
class EnvState:
    day_of_year: int # Sampled at reset()

    chargers_state: ChargersState
    battery_state: StationBattery = StationBattery()
    timestep: int = 0

    # Reward variables
    profit: float = 0.0
    uncharged_percentages: float = 0.0
    charged_overtime: int = 0 # Minutes over the desired charge time
    charged_undertime: int = 0 # Minutes under the desired charge time (positive reward)
    rejected_customers: int = 0
    left_customers: int = 0
    exceeded_capacity: float = 0.0

