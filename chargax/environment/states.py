import chex
from typing import Union, List, Optional, Literal
import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx
from dataclasses import replace

class ChargingStation(eqx.Module):
    """
        The hub is the top level of the charger hierarchy.
        It contains the topology of the chargers as a tree of ChargerNodes,
        where the root node is the grid connection.
        As well as a single object containing the state of all chargers.
    """
    charger_layout: 'StationSplitter'

    def __init__(self, num_chargers: int = 10, num_chargers_per_group: int = 2):
        assert num_chargers % num_chargers_per_group == 0, "Chargers must be divisible by chargers_per_group"
        assert num_chargers_per_group >= 1, "Chargers per group must be greater than 0"
        assert num_chargers > num_chargers_per_group, "Chargers must be greater than chargers_per_group"

        default_charger_max_rate = 100.0

        charger_indices = np.arange(num_chargers)
        charger_indices = charger_indices.reshape(-1, num_chargers_per_group)

        EVSEs = [
            StationEVSE(
                connections=ci,
                # group_capacity_max_kwh=default_charger_max_rate
            ) for ci in charger_indices
        ]
        
        combined_total_capacity = sum([group.group_capacity_max_kwh for group in EVSEs])
        grid_connection_node = StationSplitter(
            connections=EVSEs,
            group_capacity_max_kwh=combined_total_capacity
        )

        self.charger_layout = grid_connection_node

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
    def evses(self) -> List['StationEVSE']:
        return self.charger_layout.evse_children
    
    @property
    def evse_per_chargepoint(self) -> List['StationEVSE']:
        return self.charger_layout.evse_per_chargepoint_children
    
    @property
    def efficiency_per_charger(self) -> jnp.ndarray:
        efficiency_to_chargepoint = np.ones(self.num_chargers)
        for path, charger_ids in jax.tree.leaves_with_path(self.charger_layout):
            curr_node = self.charger_layout
            for node in path[:-1]: # omit the last node which is the charger
                if isinstance(node, jax.tree_util.GetAttrKey):
                    curr_node = curr_node.connections
                elif isinstance(node, jax.tree_util.SequenceKey):
                    curr_node = curr_node[node.idx]
                    efficiency_to_chargepoint[charger_ids] *= (
                        curr_node.efficiency
                    )
                    
        efficiency_to_chargepoint *= self.charger_layout.efficiency

        return efficiency_to_chargepoint

    def total_power_output(self, charger_state: 'ChargersState') -> float:
        return jnp.sum(charger_state.charger_current_now * self.voltages)
    
    def total_power_draw(self, charger_state: 'ChargersState') -> float:
        return self.total_power_output(charger_state) / self.efficiency_per_charger
    
    def total_power_loss(self, charger_state: 'ChargersState') -> float:
        return self.total_power_draw(charger_state) - self.total_power_output(charger_state)
    
    def normalize_currents(self, currents: jnp.ndarray, charger_state: 'ChargersState') -> jnp.ndarray:
        # NOTE: I am not sure if jax.tree_leaves will flatten the tree in the correct order
        splitters_and_evses: List[StationSplitter] = jax.tree_leaves(self.charger_layout, is_leaf=lambda x: isinstance(x, StationSplitter))
        for node in splitters_and_evses[::-1]:
            total_power_draw_on_group = jnp.sum(
                currents[node.charger_ids_children] * self.voltages[node.charger_ids_children]
            )
            normalization_factor = jax.lax.select(
                total_power_draw_on_group > node.group_capacity_max_kwh,
                node.group_capacity_max_kwh / total_power_draw_on_group,
                1.
            )
            currents = currents.at[node.charger_ids_children].set(
                currents[node.charger_ids_children] * normalization_factor
            )
        return currents
                    # if self.renormalize_power_levels:
        #     for i, charger_group in enumerate(idx_per_group):
        #         total_power_draw_on_group = jnp.sum(power_levels[charger_group])
        #         normalization_factor = jax.lax.select(
        #             total_power_draw_on_group > max_powers_per_group[i],
        #             max_powers_per_group[i] / total_power_draw_on_group,
        #             1.
        #         )
        #         power_levels = power_levels.at[charger_group].set(
        #             power_levels[charger_group] * normalization_factor
        #         )


        # First get EVSEs and normalize them
        # Then get list of Splitter nodes and loop over them backwards
        # and normalize the children each time

class ChargersState(eqx.Module):
    # Car variables
    # car_time_charging: chex.Array
    car_time_till_leave: chex.Array
    car_battery_now: chex.Array
    car_battery_capacity: chex.Array

    car_ac_absolute_max_charge_rate: chex.Array
    car_ac_optimal_charge_threshold: chex.Array
    car_dc_absolute_max_charge_rate: chex.Array
    car_dc_optimal_charge_threshold: chex.Array
    
    # Charger variables
    charger_current_now: chex.Array
    charger_is_car_connected: chex.Array
    charger_voltage: np.ndarray # We assume this is constant for all chargers
    # charger_current_max: chex.Array

    def __init__(
            self, 
            station: ChargingStation, 
            sample_method: Literal["empty", "random", "eu", "us"] = "empty",
            key: Optional[chex.PRNGKey] = None,
            **kwargs
        ):
        if kwargs: # To allow for replace(..., **kwargs)
            self.__dict__.update(kwargs)
            return
        
        num_chargers = station.num_chargers
        rated_voltages = [np.array([evse.voltage_rated for _ in len(evse.connections)]) for evse in station.evses]
        voltages_per_charger = np.concatenate(rated_voltages)

        self.charger_current_now = jnp.zeros(num_chargers)
        self.charger_is_car_connected = jnp.zeros(num_chargers, dtype=bool)
        self.charger_voltage = voltages_per_charger

        if sample_method == "empty":
            self.car_time_till_leave = jnp.zeros(num_chargers, dtype=int)
            self.car_battery_now = jnp.zeros(num_chargers)
            self.car_ac_absolute_max_charge_rate = jnp.zeros(num_chargers)
            self.car_ac_optimal_charge_threshold = jnp.zeros(num_chargers)
            self.car_dc_absolute_max_charge_rate = jnp.zeros(num_chargers)
            self.car_dc_optimal_charge_threshold = jnp.zeros(num_chargers)
            self.car_battery_capacity = jnp.ones(num_chargers), # ones
        
        elif sample_method == "random":
            assert key is not None, "Key must be provided for random sampling"
            keys = jax.random.split(key, 7)
            self.car_time_till_leave = jax.random.randint(keys[0], (num_chargers,), 130, 140)
            self.car_battery_now = jax.random.uniform(keys[1], (num_chargers,), minval=0.10, maxval=0.11)
            self.car_ac_absolute_max_charge_rate = jax.random.uniform(keys[2], (num_chargers,), minval=100., maxval=100.)
            self.car_ac_optimal_charge_threshold = jax.random.uniform(keys[3], (num_chargers,), minval=0.8, maxval=0.8)
            self.car_dc_absolute_max_charge_rate = jax.random.uniform(keys[4], (num_chargers,), minval=100., maxval=100.)
            self.car_dc_optimal_charge_threshold = jax.random.uniform(keys[5], (num_chargers,), minval=0.8, maxval=0.8)
            self.car_battery_capacity = jax.random.uniform(keys[6], (num_chargers,), minval=290., maxval=299.)

        elif sample_method == "eu" or sample_method == "us":
            raise NotImplementedError("Not implemented yet")
        


    @property
    def car_battery_level(self) -> jnp.ndarray:
        return self.car_battery_now / self.car_battery_capacity
    
    @property
    def charger_output_now_kwh(self) -> jnp.ndarray:
        return (self.charger_voltage * self.charger_current_now) / 1000.0

    @property
    def num_chargers(self) -> int:
        return len(self.car_battery_now)

    @property
    def car_max_charge_rate(self) -> jnp.ndarray:
        """ The current maximum charge rate of the car based on the battery level and other factors """
        # if self.charger_voltage * self.charger_current_max > 50.0: # DC charging
        #     tau = self.car_dc_optimal_charge_threshold
        #     abs_max_rate = self.car_dc_absolute_max_charge_rate
        # else: # AC charging
        tau = self.car_ac_optimal_charge_threshold
        abs_max_rate = self.car_ac_absolute_max_charge_rate
        
        # if battery_level > threshold: then linearly decay the charge rate to 5% of the max rate till 100%
        # else: charge at the max rate
        charge_rate = jnp.where(
            self.car_battery_level > tau,
            abs_max_rate * (1 - (self.car_battery_level - tau) / (1 - tau) + 0.10),
            abs_max_rate
        ) * self.charger_is_car_connected # charge rate is 0 if car is not connected
        charge_rate = jnp.clip(charge_rate, 0, abs_max_rate)
        return charge_rate
    
    # def charged_past_timestep(self, minutes_per_timestep: int) -> chex.Array:
    #     return self.charger_charge_rate_now / 60 * minutes_per_timestep

    @classmethod
    def __init__empty__(cls, num_chargers: int, voltages: Union[float, List[float]] = 230.0):
        return cls(
            car_time_till_leave = jnp.zeros(num_chargers, dtype=int),
            car_battery_now = jnp.zeros(num_chargers),
            car_battery_capacity = jnp.ones(num_chargers), # ones
            car_ac_absolute_max_charge_rate = jnp.zeros(num_chargers),
            car_ac_optimal_charge_threshold = jnp.zeros(num_chargers),
            car_dc_absolute_max_charge_rate = jnp.zeros(num_chargers), 
            car_dc_optimal_charge_threshold = jnp.zeros(num_chargers),
            charger_current_now = jnp.zeros(num_chargers),
            charger_is_car_connected = jnp.zeros(num_chargers, dtype=bool),
            charger_voltage = voltages
        )
    
    @classmethod
    def __init__filled__(cls, num_chargers: int, key: chex.PRNGKey):
        keys = jax.random.split(key, 5)
        departure_times = jax.random.randint(keys[0], (num_chargers,), 130, 140)
        arrival_battery_levels = jax.random.uniform(keys[1], (num_chargers,), minval=0.10, maxval=0.11)
        car_battery_capacitys = jax.random.uniform(keys[2], (num_chargers,), minval=290., maxval=299.)
        battery_current_levels = car_battery_capacitys * arrival_battery_levels
        optimal_charge_thresholds = jax.random.uniform(keys[3], (num_chargers,), minval=0.8, maxval=0.8)
        
        return replace(
            cls.__init__empty__(num_chargers),
            car_ac_absolute_max_charge_rate=jnp.full(num_chargers, 100.0),
            car_dc_absolute_max_charge_rate=jnp.full(num_chargers, 100.0),
            car_ac_optimal_charge_threshold=optimal_charge_thresholds,
            car_dc_optimal_charge_threshold=optimal_charge_thresholds,
            car_time_till_leave=departure_times,
            car_battery_now=battery_current_levels,
            car_battery_capacity=car_battery_capacitys,
        )
    
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
    group_capacity_max_kwh: float = eqx.field(static=True)
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
    def evse_children(self) -> List['StationEVSE']:
        return jax.tree_leaves(self.connections, is_leaf=lambda x: isinstance(x, StationEVSE))

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
    voltage_rated: float = eqx.field(static=True, default=230.0)
    current_max: float = eqx.field(static=True, default=50.0)

    def __init__(self, connections: np.ndarray, **kwargs):
        self.__dict__.update(kwargs)
        self.connections = connections
        self.group_capacity_max_kwh = (self.voltage_rated * self.current_max) / 1000.0

class StationBattery(eqx.Module):
    """
    A battery for the hub. Can be used to store excess energy or to provide energy to the grid.
    """
    capacity: float = 1000.0
    current_charge: float = 0.0
    max_charge_rate: float = 100.0
    max_discharge_rate: float = 100.0

@chex.dataclass(frozen=True)
class EnvState:
    chargers_state: ChargersState
    timestep: int = 0
    cars_leaving_satisfied: int = 0
    cars_leaving_unsatisfied: int = 0
    profit: float = 0.0



# class Chargers(eqx.Module):
#     # Car variables
#     car_rate_absolute_max: chex.Array 
#     time_waiting: chex.Array
#     time_charging: chex.Array
#     time_till_leave: chex.Array
#     battery_current: chex.Array
#     battery_max: chex.Array
#     battery_temperature: chex.Array
#     optimal_charge_threshold: chex.Array

#     # Charger variables
#     charger_rate_current: chex.Array
#     car_connected: chex.Array
#     # charger_rate_max: float # MAX RATE IS DEFINED BY THE GROUP 

#     def __init__(self, num_chargers = None, **kwargs):
#         if num_chargers:
#             self.car_rate_absolute_max = jnp.zeros(num_chargers)
#             self.time_waiting = jnp.zeros(num_chargers, dtype=int)
#             self.time_charging = jnp.zeros(num_chargers, dtype=int)
#             self.time_till_leave = jnp.zeros(num_chargers, dtype=int)
#             self.battery_current = jnp.zeros(num_chargers)
#             self.battery_max = jnp.ones(num_chargers)
#             self.battery_temperature = jnp.zeros(num_chargers)
#             self.charger_rate_current = jnp.zeros(num_chargers)
#             self.car_connected = jnp.zeros(num_chargers, dtype=bool)
#             self.optimal_charge_threshold = jnp.zeros(num_chargers)
#         else: # To allow for replace(..., **kwargs)
#             self.__dict__.update(kwargs)

#     @property
#     def battery_level(self) -> jnp.ndarray:
#         return self.battery_current / self.battery_max

#     @property
#     def num_chargers(self) -> int:
#         return len(self.car_rate_max)

#     @property
#     def car_rate_max(self) -> jnp.ndarray:
#         """ The current maximum charge rate of the car based on the battery level and other factors """
#         threshold = self.optimal_charge_threshold
#         # if battery_level > threshold: then linearly decay the charge rate to 5% of the max rate till 100%
#         # else: charge at the max rate
#         charge_rate = jnp.where(
#             self.battery_level > threshold,
#             self.car_rate_absolute_max * (1 - (self.battery_level - threshold) / (1 - threshold) + 0.10),
#             self.car_rate_absolute_max
#         ) * self.car_connected # charge rate is 0 if car is not connected
#         charge_rate = jnp.clip(charge_rate, 0, self.car_rate_absolute_max)
#         return charge_rate
    
#     def charged_past_timestep(self, minutes_per_timestep: int) -> chex.Array:
#         return self.charger_rate_current / 60 * minutes_per_timestep

#     # @classmethod
#     # def init_empty(cls, num_chargers: int):
#     #     return cls(num_chargers)
#     #     return cls(
#     #         car_rate_absolute_max=jnp.zeros(num_chargers),
#     #         time_waiting=jnp.zeros(num_chargers, dtype=int),
#     #         time_charging=jnp.zeros(num_chargers, dtype=int),
#     #         time_till_leave=jnp.zeros(num_chargers, dtype=int),
#     #         battery_current=jnp.zeros(num_chargers),
#     #         battery_max=jnp.ones(num_chargers), 
#     #         battery_temperature=jnp.zeros(num_chargers),
#     #         charger_rate_current=jnp.zeros(num_chargers),
#     #         car_connected=jnp.zeros(num_chargers, dtype=bool),
#     #         optimal_charge_threshold=jnp.zeros(num_chargers)
#     #     )
    
#     @classmethod
#     def init_custom(cls, num_chargers: int, key: chex.PRNGKey):
#         keys = jax.random.split(key, 5)
#         departure_times = jax.random.randint(keys[0], (num_chargers,), 130, 140)
#         arrival_battery_levels = jax.random.uniform(keys[1], (num_chargers,), minval=0.10, maxval=0.11)
#         car_battery_capacitys = jax.random.uniform(keys[2], (num_chargers,), minval=290., maxval=299.)
#         battery_current_levels = car_battery_capacitys * arrival_battery_levels
#         optimal_charge_thresholds = jax.random.uniform(keys[3], (num_chargers,), minval=0.8, maxval=0.8)
        
#         return replace(
#             cls(num_chargers),
#             car_rate_absolute_max=jnp.full(num_chargers, 100.0),
#             time_till_leave=departure_times,
#             battery_current=battery_current_levels,
#             battery_max=car_battery_capacitys,
#             battery_temperature=jnp.full(num_chargers, 35.0),
#             optimal_charge_threshold=optimal_charge_thresholds
#         )
    
# class ChargerGroup(eqx.Module):
#     """
#     A group can contain:
#     - Charger indices (list of integers referencing chargers in `Chargers`)
#     - Other ChargerGroups
#     """
#     connections: List[Union[int, 'ChargerGroup']]
#     group_capacity_max_kwh: float = eqx.field(static=True)
#     efficiency: float = eqx.field(static=True, default=0.995)

#     def group_rate_current(self, chargers: Chargers) -> float:
#         return jnp.sum(chargers.charger_rate_current[self.charger_idx_in_group])
        
#     def total_draw_group(self, chargers: Chargers) -> float:
#         draw = self.group_rate_current(chargers)
#         loss = self.efficiency ** self.num_parents_per_charger
#         required_total_draw = draw / loss
#         return required_total_draw
    
#     @property
#     def num_parents_per_charger(self) -> float:
#         # For each node, we need to know how many parents it has
#         nodes_w_path = jax.tree.leaves_with_path(self.connections)
#         parents_per_charger = np.zeros(self.number_of_chargers_in_group)
#         index = 0
#         for path, nodes in nodes_w_path:
#             num_parents = (len(path) // 3) + 1 # +1 for the root node
#             num_nodes_w_parents = len(nodes)
#             parents_per_charger[index:index+num_nodes_w_parents] = num_parents
#             index += num_nodes_w_parents

#         return parents_per_charger
    
#     @property
#     def chargers_in_group(self): 
#         return jax.tree_leaves(self.connections)
    
#     @property
#     def charger_idx_in_group(self) -> chex.Array:
#         return np.concatenate(self.chargers_in_group)
    
#     @property
#     def number_of_chargers_in_group(self) -> int:
#         return len(self.charger_idx_in_group)
    
#     @property
#     def bottom_level(self) -> List['ChargerGroup']:
#         """ 
#             Returns a list of groups that are the final connections
#             meaning their connections are charger indices instead of subgroups
#         """
#         if isinstance(self.connections[0], ChargerGroup):
#             return [group.bottom_level for group in self.connections]
#         else:
#             return self
        
#     def get_parent(self, item: Union[np.ndarray, 'ChargerGroup']) -> 'ChargerGroup':
#         """ within connections, find the item that has "item" as a direct child under connections """
#         for conn in self.connections:
#             if conn is item:
#                 return self
#             elif isinstance(conn, ChargerGroup):
#                 parent = conn.get_parent(item)
#                 if parent:
#                     return parent
#         return None
    

# class ChargerGroup(eqx.Module):
#     """
#         This could be a list of chargers or a list of 
#         chargergroups (representing switch boards or transformers)
#     """
#     group_capacity_max_kwh: float = eqx.field(static=True)
#     connections: List[Union[ChargersState, 'ChargerGroup']]

#     @property
#     def group_capacity_current(self) -> float:
#         connected_chargers = jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, ChargersState))
#         return jnp.sum(jnp.array([charger.charger_rate_current for charger in connected_chargers]))
    
#     @property
#     def chargers_in_group(self) -> List[ChargersState]:
#         return jax.tree_leaves(self.connections, is_leaf=lambda x: isinstance(x, ChargersState))
    
#     @property
#     def number_of_chargers_in_group(self) -> int:
#         return len(self.chargers_in_group)

# @chex.dataclass(frozen=True)
# class ExogenuousState:
    
#     pass

#     # weather
#     # time of day
#     # time of year
#     # holidays
#     # elec price

# @chex.dataclass(frozen=True)
# class EnvState:
#     chargers: Chargers
#     timestep: int = 0
#     cars_leaving_satisfied: int = 0
#     cars_leaving_unsatisfied: int = 0
#     profit: float = 0.0
