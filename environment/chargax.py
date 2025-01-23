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
    chargers: Chargers
    arrival_distributions: List[distrax.Distribution]

    def __post_init__(self):
        # self.__setattr__("some_property", some_property_value)
        pass

    def __check_init__(self):
        # assert anything ...
        pass 

    def reset_env(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        state = EnvState(
            grid_connection=self.charger_topology,
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
            grid_connection=new_state.grid_connection,
            timestep=state.timestep + 1
        )
    
        return timestep_object, new_state
    
    @jax.jit
    def clear_out_current_cars(self, state: EnvState) -> EnvState:

        breakpoint()
        time_till_leaves = jnp.array(
            jax.tree.map(lambda x: x.car.time_till_leave, state.grid_connection.chargers_in_group, is_leaf=lambda x: isinstance(x, ChargerState))
        )
        time_till_leaves = jnp.maximum(time_till_leaves - 5, 0)
        # create an array filled with F

        # Put back in original position:
        charging_stations = jax.tree.map(
            lambda charger, ttl: replace(
                charger, car=replace(
                    charger.car, time_till_leave=ttl
                )
            ),
            state.grid_connection.chargers_in_group,
            time_till_leaves,
            is_leaf=lambda x: isinstance(x, ChargerState)
        )



        # charging_stations = jax.tree.map(
        #     lambda charger: replace(
        #         charger, car=replace(
        #             charger.car, 
        #             time_till_leave=jnp.maximum(charger.car.time_till_leave - 5, 0)
        #         )
        #     ),
        #     state.grid_connection,
        #     is_leaf=lambda x: isinstance(x, ChargerState)
        # )

        return replace(state, grid_connection=charging_stations)
    
    @jax.jit
    def add_new_cars(self, state: EnvState, key: chex.PRNGKey) -> EnvState:

        # We need some trickery here to remain compatible with Jit and Vmap

        def sample_car(key: chex.PRNGKey) -> CarState:
            keys = jax.random.split(key, 5)
            departure_time = jax.random.randint(keys[0], (), 10, 60)
            arrival_battery_level = jax.random.uniform(keys[1], (), minval=3., maxval=50.)
            car_battery_capacity = jax.random.uniform(keys[2], (), minval=200., maxval=300.)
            return CarState(
                time_till_leave=departure_time,
                battery_level=arrival_battery_level,
                battery_capacity=car_battery_capacity,
                battery_temperature=35.0,
                car_rate_max=75.0
            )
        
        # TODO: Can't index with a tracer into a list; 
        # Some inefficient trick with a lax.switch exists, but we'll need something better
        new_cars_amount = self.arrival_distributions[0].sample(seed=key)
        new_cars_amount = jnp.maximum(new_cars_amount, 1).astype(int) # Tmp: at least one car
        
        # First create a boolean lists indicating where the new arriving cars should be placed
        # This sets the index to False when a car is already connected
        # Or when car is not arriving (we only set new_cars_amount of True values)
        # Cars will just take the first available spot 
        not_connected_chargers = jax.tree.map(
            lambda x: jnp.logical_not(x.car_connected),
            state.grid_connection.chargers_in_group,
            is_leaf=lambda x: isinstance(x, ChargerState)
        )
        not_connected_chargers = jnp.array(not_connected_chargers)
        sort_order = jnp.argsort(not_connected_chargers, descending=True)
        required_chargers = jnp.arange(state.grid_connection.number_of_chargers_in_group) < new_cars_amount
        required_chargers = jnp.zeros_like(required_chargers).at[sort_order].set(required_chargers)
        required_chargers = [c for c in required_chargers]

        
        # Sample a car for every possible charging spot (even if it's not needed -- to stay compatible with vmap)
        # And set car_connected to False if the spot is not needed based on the previously created required_chargers
        new_car_keys = jax.random.split(key, state.grid_connection.number_of_chargers_in_group)
        new_chargers = [
            ChargerState(
                charger_rate_current=charger.charger_rate_current,
                car_connected=True,
                car=sample_car(new_car_keys[i])
            ) for i, charger in enumerate(state.grid_connection.chargers_in_group)
        ]
        new_chargers = jax.tree.map(
            lambda x, y: ChargerState(
                charger_rate_current=x.charger_rate_current,
                car_connected=y,
                car=x.car
            ),
            new_chargers,
            required_chargers,
            is_leaf=lambda x: isinstance(x, ChargerState)
        )

        # Now merge the two lists of charging stations (and cars) together
        grid_connection_leaves, treedef = jax.tree_flatten(state.grid_connection, is_leaf=lambda x: isinstance(x, ChargerState))
        updated_chargers = jax.tree_map(
            lambda x, y: jax.lax.cond(
                x.car_connected, lambda: x, lambda: y
            ), grid_connection_leaves, new_chargers, is_leaf=lambda x: isinstance(x, ChargerState)
        )

        # Unflatten back to original pytree structure
        updated_chargers = treedef.unflatten(updated_chargers)

        return  replace(state, grid_connection=updated_chargers)

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