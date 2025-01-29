from typing import Tuple, Dict, List
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx 
from dataclasses import replace, asdict
import chex
import distrax

from chargax import JaxBaseEnv, TimeStep, EnvState, ChargerGroup, Chargers, Discrete, MultiDiscrete

# disable jit
# jax.config.update("jax_disable_jit", True)

class Chargax(JaxBaseEnv):

    charger_topology: ChargerGroup
    arrival_distributions: List[distrax.Distribution]
    num_discretization_levels: int = 20 # 10 would mean each charger can charge 10%, 20%, ... of its max rate
    minutes_per_timestep: int = 5
    renormalize_power_levels: bool = True

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
        self.charger_topology.group_rate_current(state.chargers)
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

        idx_per_group = self.charger_topology.chargers_in_group
        max_powers_per_group = jax.tree.map(lambda x: x.group_capacity_max, self.charger_topology.bottom_level, is_leaf=lambda x: isinstance(x, ChargerGroup))
        assert len(max_powers_per_group) == len(idx_per_group)
        for i, charger_group in enumerate(idx_per_group):
            power_levels = power_levels.at[charger_group].set(
                actions[charger_group] * max_powers_per_group[i]
            )

        power_levels = jnp.minimum(power_levels, state.chargers.car_rate_max)
        power_levels = power_levels * state.chargers.car_connected # Bit redundant, but it's fine
        if self.renormalize_power_levels:
            for i, charger_group in enumerate(idx_per_group):
                total_power_draw_on_group = jnp.sum(power_levels[charger_group])
                normalization_factor = jax.lax.select(
                    total_power_draw_on_group > max_powers_per_group[i],
                    max_powers_per_group[i] / total_power_draw_on_group,
                    1.
                )
                power_levels = power_levels.at[charger_group].set(
                    power_levels[charger_group] * normalization_factor
                )

        new_charger_state = replace(
            state.chargers,
            charger_rate_current=power_levels
        )  

        # TODO: Renormalize power levels per group? (this now only occurs on the lowest level of chargers)
        # NOTE: this is probably already a fair assumption though.

        return replace(
            state,
            chargers=new_charger_state
        )
    
    def update_charging_levels(self, state: EnvState) -> EnvState:
        charge_past_timestep = state.chargers.charger_rate_current / 60 * self.minutes_per_timestep
        new_battery = jnp.minimum(
            state.chargers.battery_current + charge_past_timestep,
            state.chargers.battery_max
        )

        return replace(
            state,
            chargers=replace(
                state.chargers,
                battery_current=new_battery
            )
        )
    
    def update_time_and_clear_cars(self, state: EnvState) -> EnvState:

        car_waiting_times = state.chargers.time_till_leave - self.minutes_per_timestep
        car_waiting_times = jnp.maximum(car_waiting_times, 0)
        car_leaving = jnp.logical_and(car_waiting_times == 0, state.chargers.car_connected)
        cars_leaving_satisfied = jnp.logical_and(
            car_leaving, 
            state.chargers.battery_level >= 0.8
        ).sum()
        car_connected = (car_waiting_times * state.chargers.car_connected).astype(bool)
        chargers = replace(
            state.chargers,
            time_till_leave=car_waiting_times,
            car_connected=car_connected,
        )
        return replace(state, 
            chargers=chargers,
            cars_leaving_satisfied=state.cars_leaving_satisfied + cars_leaving_satisfied
        )
    
    def add_new_cars(self, state: EnvState, key: chex.PRNGKey) -> EnvState:

        # TODO: Can't index with a tracer into a list; 
        # Some inefficient trick with a lax.switch exists, but we'll need something better
        # def sample_distribution(i, key: chex.PRNGKey) -> int:
        #     return self.arrival_distributions[i].sample(seed=key)
        # sample_distribution_fn_seq = [partial(sample_distribution, i, key) for i in range(len(self.arrival_distributions))]
        # new_cars_amount = jax.lax.switch(
        #     state.timestep, sample_distribution_fn_seq
        # )
        # new_cars_amount = self.arrival_distributions[0].sample(seed=key)
        # new_cars_amount = jnp.maximum(new_cars_amount, 1).astype(int) # Tmp: at least one car

        # Probably create this on reset()
        ev_arrival_means = [self.arrival_distributions[i]._loc for i in range(len(self.arrival_distributions))]
        ev_arrival_stds = [self.arrival_distributions[i]._scale for i in range(len(self.arrival_distributions))]
        ev_arrival_means = jnp.array(ev_arrival_means)
        ev_arrival_stds = jnp.array(ev_arrival_stds)
        new_cars_amount = jax.random.normal(key, ()) * ev_arrival_stds[state.timestep] + ev_arrival_means[state.timestep]


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
        return new_state.cars_leaving_satisfied - old_state.cars_leaving_satisfied
        
    def get_terminated(self, state: EnvState) -> bool:
        return False
    
    def get_truncated(self, state: EnvState) -> bool:
        return state.timestep >= len(self.arrival_distributions)
    
    def get_info(self, state: EnvState, actions) -> Dict[str, chex.Array]:
        return {}
    
    def observation_space(self, agent: str):
        raise NotImplementedError()
    
    @property
    def action_space(self):
        return MultiDiscrete(
            np.full(self.charger_topology.number_of_chargers_in_group, self.num_discretization_levels)
        )

if __name__ == "__main__":
    env = Chargax()
    print(env.name)
    key = jax.random.PRNGKey(0)
    env.reset(key)