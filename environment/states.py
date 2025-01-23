import chex
from typing import Union, List
import jax.numpy as jnp
import equinox as eqx

@chex.dataclass(frozen=True)
class Chargers:
    # Car variables
    car_rate_max: chex.Array
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
    def number_of_chargers(self) -> int:
        return len(self.car_rate_max)

    @property
    def car_charge_curve(self) -> jnp.ndarray:
        return jnp.array([self.battery_level, self.battery_capacity, self.battery_temperature])

    @classmethod
    def instantiate(cls, num_chargers: int):
        return cls(
            car_rate_max=jnp.zeros(num_chargers),
            time_waiting=jnp.zeros(num_chargers, dtype=int),
            time_charging=jnp.zeros(num_chargers, dtype=int),
            time_till_leave=jnp.zeros(num_chargers, dtype=int),
            battery_level=jnp.zeros(num_chargers),
            battery_capacity=jnp.zeros(num_chargers),
            battery_temperature=jnp.zeros(num_chargers),
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
        if isinstance(self.connections[0], ChargerGroup): # Chargers
            return jnp.sum(jnp.array([group.group_capacity_current(chargers) for group in self.connections]))
        else:
            return jnp.sum(chargers.charger_rate_current[self.connections[0]])
            
    

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
    grid_connection: ChargerGroup
    timestep: int = 0
