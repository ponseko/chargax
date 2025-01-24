from typing import Tuple, Dict, List
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx 
from dataclasses import replace, asdict
import chex
import distrax

from environment.base_and_wrappers import JaxBaseEnv, TimeStep
from environment.states import EnvState, ChargerGroup, Chargers
from environment.spaces import Discrete, MultiDiscrete

# disable jit
# jax.config.update("jax_disable_jit", True)

class Chargax(JaxBaseEnv):

    charger_topology: ChargerGroup
    arrival_distributions: List[distrax.Distribution]
    num_discretization_levels: int = 20 # 10 would mean each charger can charge 10%, 20%, ... of its max rate

    def __post_init__(self):
        # self.__setattr__("some_property", some_property_value)
        pass

    def __check_init__(self):
        # assert anything ...
        pass 

    def reset_env(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        state = EnvState(
            chargers=Chargers.init_empty(self.charger_topology.number_of_chargers_in_group),
        )
        observation = self.get_observations(state)
        return observation, state
    
    def step_env(self, rng: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[TimeStep, EnvState]:

        rng, key = jax.random.split(rng)
        new_state = state

        new_state = self.set_power_levels(new_state, actions)
        new_state = self.update_charging_levels(new_state)

        new_state = self.update_time_and_clear_cars(new_state)
        new_state = self.add_new_cars(new_state, key)

        timestep_object = TimeStep(
            observation=self.get_observations(state),
            reward=self.get_rewards(state, new_state),
            terminated=self.get_terminated(state),
            truncated=self.get_truncated(state),
            info=self.get_info(state, actions)
        )

        new_state = EnvState(
            chargers=new_state.chargers,
            timestep=state.timestep + 1
        )
    
        return timestep_object, new_state
    
    def set_power_levels(self, state: EnvState, actions: chex.Array) -> EnvState:

        actions = actions / self.num_discretization_levels
        power_levels = jnp.zeros(self.charger_topology.number_of_chargers_in_group)

        charger_idx = np.arange(len(actions))
        bottom_level = self.charger_topology.bottom_level
        slices = jax.tree.map(lambda x: len(x.connections[0]), bottom_level, is_leaf=lambda x: isinstance(x, ChargerGroup))
        slice_ids = np.cumsum(slices)[:-1]
        charger_idx_per_slice = np.split(charger_idx, slice_ids)
        max_group_powers = jax.tree.map(lambda x: x.group_capacity_max, bottom_level, is_leaf=lambda x: isinstance(x, ChargerGroup))
        assert len(max_group_powers) == len(charger_idx_per_slice)
        for i in range(len(charger_idx_per_slice)):
            power_levels = power_levels.at[charger_idx_per_slice[i]].set(
                actions[charger_idx_per_slice[i]] * max_group_powers[i]
            )

        power_levels = jnp.minimum(power_levels, state.chargers.car_rate_max)
        power_levels = power_levels * state.chargers.car_connected

        new_charger_state = replace(
            state.chargers,
            charger_rate_current=power_levels
        )

        # TODO: We may want to renormalize the power levels here at each group level IF the sum exceeds the max group capacity
        # NOTE: Is that realisitc? 
        # Or should this just be a negative reward ?
        # Probably a real controller would have this as a hard contraint... ?
        # Probably just make this an optional feature for the environment -- toggleable by a flag

        return replace(
            state,
            chargers=new_charger_state
        )
    
    def update_charging_levels(self, state: EnvState) -> EnvState:
        new_battery_level = state.chargers.battery_level + state.chargers.charger_rate_current
        new_battery_level = jnp.minimum(new_battery_level, state.chargers.battery_capacity)

        return replace(
            state,
            chargers=replace(
                state.chargers,
                battery_level=new_battery_level
            )
        )
    
    def update_time_and_clear_cars(self, state: EnvState) -> EnvState:

        car_waiting_times = state.chargers.time_till_leave - 5
        car_waiting_times = jnp.maximum(car_waiting_times, 0)
        car_leaving = jnp.logical_and(car_waiting_times == 0, state.chargers.car_connected)
        cars_leaving_satisfied = jnp.logical_and(
            car_leaving, 
            state.chargers.battery_level >= (0.9 * state.chargers.battery_capacity)
        ).sum()
        car_connected = (car_waiting_times * state.chargers.car_connected).astype(bool)
        chargers = replace(
            state.chargers,
            time_till_leave=car_waiting_times,
            car_connected=car_connected,
        )
        jax.debug.breakpoint()
        return replace(state, 
            chargers=chargers,
            cars_leaving_satisfied=state.cars_leaving_satisfied + cars_leaving_satisfied
        )
    
    def add_new_cars(self, state: EnvState, key: chex.PRNGKey) -> EnvState:

        # TODO: Can't index with a tracer into a list; 
        # Some inefficient trick with a lax.switch exists, but we'll need something better
        new_cars_amount = self.arrival_distributions[0].sample(seed=key)
        new_cars_amount = jnp.maximum(new_cars_amount, 1).astype(int) # Tmp: at least one car

        # Generate new chargers and put the car_connected to False when:
        # 1. The index of the charger is already connected to a car
        # 2. There are less incoming cars than chargers
        state.chargers.car_connected
        not_connected_chargers = jnp.logical_not(state.chargers.car_connected)
        sort_order = jnp.argsort(not_connected_chargers, descending=True)
        required_chargers = jnp.arange(state.chargers.num_chargers) < new_cars_amount
        required_chargers_in_order = jnp.zeros_like(required_chargers).at[sort_order].set(required_chargers)
        incoming_chargers = Chargers.init_custom(
            len(self.charger_topology.charger_idx_in_group),
            key
        )
        incoming_chargers = replace(
            incoming_chargers, 
            car_connected=required_chargers_in_order,
        )

        # Merge the incoming chargers with the current chargers
        merged_chargers = jax.tree_map(
            lambda new, curr: jax.lax.select(
                required_chargers_in_order, new, curr
            ), incoming_chargers, state.chargers
        )

        return replace(state, chargers=merged_chargers)

    def get_observations(self, state: EnvState) -> Dict[str, chex.Array]:
        return jnp.array([0,0])

    def get_action_masks(self, state: EnvState) -> chex.Array:
        raise NotImplementedError()

    def get_rewards(self, old_state: EnvState, new_state: EnvState) -> chex.Array:
        return new_state.cars_leaving_satisfied
        
    def get_terminated(self, state: EnvState) -> bool:
        return False
    
    def get_truncated(self, state: EnvState) -> bool:
        return False
    
    def get_info(self, state: EnvState, actions) -> Dict[str, chex.Array]:
        return {}
    
    def observation_space(self, agent: str):
        raise NotImplementedError()
    
    @property
    def action_space(self):
        return MultiDiscrete(
            np.full(self.charger_topology.number_of_chargers_in_group, 10)
        )

if __name__ == "__main__":
    env = Chargax()
    print(env.name)
    key = jax.random.PRNGKey(0)
    env.reset(key)