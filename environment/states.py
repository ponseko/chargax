import chex
from typing import Union, List
import jax.numpy as jnp
import jax
import equinox as eqx

@chex.dataclass(frozen=True)
class Chargers:
    # Car variables
    car_rate_absolute_max: chex.Array 
    time_waiting: chex.Array
    time_charging: chex.Array
    time_till_leave: chex.Array
    battery_level: chex.Array
    battery_capacity: chex.Array
    battery_temperature: chex.Array

    # Charger variables
    charger_rate_current: chex.Array
    car_connected: chex.Array
    # charger_rate_max: float # MAX RATE IS DEFINED BY THE GROUP 

    @property
    def num_chargers(self) -> int:
        return len(self.car_rate_max)

    @property
    def car_rate_max(self) -> jnp.ndarray:
        """ The current maximum charge rate of the car based on the battery level and other factors """
        # For now, just return the absolute max (if connected)
        return self.car_rate_absolute_max * self.car_connected

    @classmethod
    def init_empty(cls, num_chargers: int):
        return cls(
            car_rate_absolute_max=jnp.zeros(num_chargers),
            time_waiting=jnp.zeros(num_chargers, dtype=int),
            time_charging=jnp.zeros(num_chargers, dtype=int),
            time_till_leave=jnp.zeros(num_chargers, dtype=int),
            battery_level=jnp.zeros(num_chargers),
            battery_capacity=jnp.zeros(num_chargers),
            battery_temperature=jnp.zeros(num_chargers),
            charger_rate_current=jnp.zeros(num_chargers),
            car_connected=jnp.zeros(num_chargers, dtype=bool)
        )
    
    @classmethod
    def init_custom(cls, num_chargers: int, key: chex.PRNGKey):
        keys = jax.random.split(key, 5)
        departure_times = jax.random.randint(keys[0], (num_chargers,), 10, 60)
        arrival_battery_levels = jax.random.uniform(keys[1], (num_chargers,), minval=3., maxval=50.)
        car_battery_capacitys = jax.random.uniform(keys[2], (num_chargers,), minval=200., maxval=300.)
        return cls(
            car_rate_absolute_max=jnp.full(num_chargers, 75.0),
            time_waiting=jnp.zeros(num_chargers, dtype=int),
            time_charging=jnp.zeros(num_chargers, dtype=int),
            time_till_leave=departure_times,
            battery_level=arrival_battery_levels,
            battery_capacity=car_battery_capacitys,
            battery_temperature=jnp.full(num_chargers, 35.0),
            charger_rate_current=jnp.zeros(num_chargers),
            car_connected=jnp.zeros(num_chargers, dtype=bool)
        )
    
class ChargerGroup(eqx.Module):
    """
    A group can contain:
    - Charger indices (list of integers referencing chargers in `Chargers`)
    - Other ChargerGroups
    """
    group_capacity_max: float = eqx.field(static=True)
    connections: List[Union[int, 'ChargerGroup']]

    def group_capacity_current(self, chargers: Chargers) -> float:
        return jnp.sum(chargers.charger_rate_current[self.charger_idx_in_group])
        # if isinstance(self.connections[0], ChargerGroup): # Chargers
        #     return jnp.sum(jnp.array([group.group_capacity_current(chargers) for group in self.connections]))
        # else:
        #     return jnp.sum(chargers.charger_rate_current[self.connections[0]])
    
    @property
    def charger_idx_in_group(self) -> chex.Array:
        return jnp.concatenate(
            jax.tree.leaves(self.connections)
        )
    
    @property
    def number_of_chargers_in_group(self) -> int:
        return len(self.charger_idx_in_group)
    
    @property
    def bottom_level(self) -> List[int]:
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
