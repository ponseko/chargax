import chex
from typing import Union, List
import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx

@chex.dataclass(frozen=True)
class Chargers:
    # Car variables
    car_rate_absolute_max: chex.Array 
    time_waiting: chex.Array
    time_charging: chex.Array
    time_till_leave: chex.Array
    battery_current: chex.Array
    battery_max: chex.Array
    battery_temperature: chex.Array
    optimal_charge_threshold: chex.Array

    # Charger variables
    charger_rate_current: chex.Array
    car_connected: chex.Array
    # charger_rate_max: float # MAX RATE IS DEFINED BY THE GROUP 

    @property
    def battery_level(self) -> jnp.ndarray:
        return self.battery_current / self.battery_max

    @property
    def num_chargers(self) -> int:
        return len(self.car_rate_max)

    @property
    def car_rate_max(self) -> jnp.ndarray:
        """ The current maximum charge rate of the car based on the battery level and other factors """
        threshold = 0.65
        # if battery_level > threshold: then linearly decay the charge rate to 5% of the max rate till 100%
        # else: charge at the max rate
        charge_rate = jnp.where(
            self.battery_level > threshold,
            self.car_rate_absolute_max * (1 - (self.battery_level - threshold) / (1 - threshold) + 0.10),
            self.car_rate_absolute_max
        ) * self.car_connected # charge rate is 0 if car is not connected
        charge_rate = jnp.clip(charge_rate, 0, self.car_rate_absolute_max)
        return charge_rate

    @classmethod
    def init_empty(cls, num_chargers: int):
        return cls(
            car_rate_absolute_max=jnp.zeros(num_chargers),
            time_waiting=jnp.zeros(num_chargers, dtype=int),
            time_charging=jnp.zeros(num_chargers, dtype=int),
            time_till_leave=jnp.zeros(num_chargers, dtype=int),
            battery_current=jnp.zeros(num_chargers),
            battery_max=jnp.ones(num_chargers), 
            battery_temperature=jnp.zeros(num_chargers),
            charger_rate_current=jnp.zeros(num_chargers),
            car_connected=jnp.zeros(num_chargers, dtype=bool),
            optimal_charge_threshold=jnp.zeros(num_chargers)
        )
    
    @classmethod
    def init_custom(cls, num_chargers: int, key: chex.PRNGKey):
        keys = jax.random.split(key, 5)
        departure_times = jax.random.randint(keys[0], (num_chargers,), 10, 60)
        arrival_battery_levels = jax.random.uniform(keys[1], (num_chargers,), minval=0.05, maxval=0.5)
        car_battery_capacitys = jax.random.uniform(keys[2], (num_chargers,), minval=200., maxval=300.)
        battery_current_levels = car_battery_capacitys * arrival_battery_levels
        optimal_charge_thresholds = jax.random.uniform(keys[3], (num_chargers,), minval=0.5, maxval=0.8)
        return cls(
            car_rate_absolute_max=jnp.full(num_chargers, 75.0),
            time_waiting=jnp.zeros(num_chargers, dtype=int),
            time_charging=jnp.zeros(num_chargers, dtype=int),
            time_till_leave=departure_times,
            battery_current=battery_current_levels,
            battery_max=car_battery_capacitys,
            battery_temperature=jnp.full(num_chargers, 35.0),
            charger_rate_current=jnp.zeros(num_chargers),
            car_connected=jnp.zeros(num_chargers, dtype=bool),
            optimal_charge_threshold=optimal_charge_thresholds
        )
    
class ChargerGroup(eqx.Module):
    """
    A group can contain:
    - Charger indices (list of integers referencing chargers in `Chargers`)
    - Other ChargerGroups
    """
    connections: List[Union[int, 'ChargerGroup']]
    group_capacity_max: float = eqx.field(static=True)
    efficiency: float = eqx.field(static=True, default=0.995)

    def group_rate_current(self, chargers: Chargers) -> float:
        return jnp.sum(chargers.charger_rate_current[self.charger_idx_in_group])
        
    def total_draw_group(self, chargers: Chargers) -> float:
        draw = self.group_rate_current(chargers)
        loss = self.efficiency ** self.num_parents_per_charger
        required_total_draw = draw / loss
        return required_total_draw
    
    @property
    def num_parents_per_charger(self) -> float:
        # For each node, we need to know how many parents it has
        nodes_w_path = jax.tree.leaves_with_path(self.connections)
        parents_per_charger = np.zeros(self.number_of_chargers_in_group)
        index = 0
        for path, nodes in nodes_w_path:
            num_parents = (len(path) // 3) + 1 # +1 for the root node
            num_nodes_w_parents = len(nodes)
            parents_per_charger[index:index+num_nodes_w_parents] = num_parents
            index += num_nodes_w_parents

        return parents_per_charger
    
    @property
    def chargers_in_group(self): 
        return jax.tree_leaves(self.connections)
    
    @property
    def charger_idx_in_group(self) -> chex.Array:
        return np.concatenate(self.chargers_in_group)
    
    @property
    def number_of_chargers_in_group(self) -> int:
        return len(self.charger_idx_in_group)
    
    @property
    def bottom_level(self) -> List['ChargerGroup']:
        """ 
            Returns a list of groups that are the final connections
            meaning their connections are charger indices instead of subgroups
        """
        if isinstance(self.connections[0], ChargerGroup):
            return [group.bottom_level for group in self.connections]
        else:
            return self
            
    

# class ChargerGroup(eqx.Module):
#     """
#         This could be a list of chargers or a list of 
#         chargergroups (representing switch boards or transformers)
#     """
#     group_capacity_max: float = eqx.field(static=True)
#     connections: List[Union[ChargerState, 'ChargerGroup']]

#     @property
#     def group_capacity_current(self) -> float:
#         connected_chargers = jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, ChargerState))
#         return jnp.sum(jnp.array([charger.charger_rate_current for charger in connected_chargers]))
    
#     @property
#     def chargers_in_group(self) -> List[ChargerState]:
#         return jax.tree_leaves(self.connections, is_leaf=lambda x: isinstance(x, ChargerState))
    
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

@chex.dataclass(frozen=True)
class EnvState:
    chargers: Chargers
    timestep: int = 0
    cars_leaving_satisfied: int = 0
