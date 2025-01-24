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


# disable jit
# jax.config.update("jax_disable_jit", True)

class Chargax(JaxBaseEnv):

    charger_topology: ChargerGroup
    arrival_distributions: List[distrax.Distribution]

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
    
    @jax.jit
    def step_env(self, rng: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[TimeStep, EnvState]:

        rng, key = jax.random.split(rng)
        new_state = state
        new_state = self.clear_out_current_cars(new_state)
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
    
    def clear_out_current_cars(self, state: EnvState) -> EnvState:

        car_waiting_times = state.chargers.time_till_leave - 5
        car_waiting_times = jnp.maximum(car_waiting_times, 0)
        car_connected = (car_waiting_times * state.chargers.car_connected).astype(bool)
        chargers = replace(
            state.chargers,
            time_till_leave=car_waiting_times,
            car_connected=car_connected,
            car_rate_max=state.chargers.car_rate_max * car_connected
        )
        return replace(state, chargers=chargers)
    
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
            car_rate_max=incoming_chargers.car_rate_max * required_chargers_in_order
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
        return 0.0
        
    def get_terminated(self, state: EnvState) -> bool:
        return False
    
    def get_truncated(self, state: EnvState) -> bool:
        return False
    
    def get_info(self, state: EnvState, actions) -> Dict[str, chex.Array]:
        return {}
    
    def observation_space(self, agent: str):
        raise NotImplementedError()
        
    def action_space(self, agent: str):
        raise NotImplementedError()

if __name__ == "__main__":
    env = Chargax()
    print(env.name)
    key = jax.random.PRNGKey(0)
    env.reset(key)