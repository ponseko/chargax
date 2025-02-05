import chex
from typing import Union, List
import jax.numpy as jnp
import jax
import equinox as eqx

@chex.dataclass(frozen=True)
class CarState:
    # Car variables
    car_rate_max: float = 0.0
    time_waiting: int = 0
    time_charging: int = 0
    time_till_leave: int = 0
    battery_level: float = 0.0
    battery_capacity: float = 0.0
    battery_temperature: float = 0.0

    @property
    def car_charge_curve(self) -> jnp.ndarray:
        return jnp.array([self.battery_level, self.battery_capacity, self.battery_temperature])

# @chex.dataclass(frozen=True)
class ChargerState(eqx.Module):
    # charger_rate_max: float # MAX RATE IS DEFINED BY THE GROUP
    charger_rate_current: float = 0.0
    car_connected: bool = False
    car: CarState = CarState() # Always defined, but connected based on "car_connected"

class ChargerGroup(eqx.Module):
    """
        This could be a list of chargers or a list of 
        chargergroups (representing switch boards or transformers)
    """
    group_capacity_max_kw: float = eqx.field(static=True)
    connections: List[Union[ChargerState, 'ChargerGroup']]

    @property
    def group_capacity_current(self) -> float:
        connected_chargers = jax.tree.leaves(self.connections, is_leaf=lambda x: isinstance(x, ChargerState))
        return jnp.sum(jnp.array([charger.charger_rate_current for charger in connected_chargers]))
    
    @property
    def chargers_in_group(self) -> List[ChargerState]:
        return jax.tree_leaves(self.connections, is_leaf=lambda x: isinstance(x, ChargerState))
    
    @property
    def number_of_chargers_in_group(self) -> int:
        return len(self.chargers_in_group)

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
