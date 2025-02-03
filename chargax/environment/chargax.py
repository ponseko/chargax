from typing import Tuple, Dict, List, Callable, Type
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx 
from dataclasses import replace, asdict
import chex

from chargax import JaxBaseEnv, TimeStep, EnvState, ChargersState, MultiDiscrete, Box, ChargingStation

# disable jit
# jax.config.update("jax_disable_jit", True)

class Chargax(JaxBaseEnv):
    
    # charger_topology: ChargerGroup
    
    ev_arrival_data_means: np.ndarray = eqx.field(converter=np.asarray)
    ev_arrival_data_stds: np.ndarray = eqx.field(converter=np.asarray)
    station: ChargingStation = ChargingStation()
    num_discretization_levels: int = 20 # 10 would mean each charger can charge 10%, 20%, ... of its max rate
    minutes_per_timestep: int = 5
    renormalize_currents: bool = True
    elec_price_data: Callable = lambda x: jnp.sin(x) + 1
    elec_sell_price: float = 1.75 # $/kWh # TODO: Make this a function of time ?

    def __post_init__(self):
        # self.__setattr__("some_property", some_property_value)
        pass

    def __check_init__(self):
        # assert anything ...
        pass 

    def reset_env(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        state = EnvState(
            chargers_state=ChargersState.__init__empty__(self.station.number_of_chargers),
        )
        observation = self.get_observation(state)
        return observation, state
    
    def step_env(self, rng: chex.PRNGKey, old_state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[TimeStep, EnvState]:

        rng, key = jax.random.split(rng)
        new_state = old_state

        self.station.total_power_loss(new_state.chargers_state)

        new_state = self.set_new_currents(new_state, actions)
        new_state = self.update_charging_levels(new_state)

        new_state = self.process_buy_and_sell_electricity(new_state)

        new_state = self.update_time_and_clear_cars(new_state)
        new_state = self.add_new_cars(new_state, key)

        new_state = replace(
            new_state,
            timestep=old_state.timestep + 1
        )

        self.charger_topology.get_parent(self.charger_topology.connections[0].connections[0])

        timestep_object = TimeStep(
            observation=self.get_observation(new_state),
            reward=self.get_reward(old_state, new_state),
            terminated=self.get_terminated(new_state),
            truncated=self.get_truncated(new_state),
            info=self.get_info(new_state, actions)
        )
    
        return timestep_object, new_state
    
    def set_new_currents(self, state: EnvState, actions: chex.Array) -> EnvState:

        actions = actions / self.num_discretization_levels
        currents = jnp.zeros(self.station.number_of_chargers)

        idx_per_group = self.station.grouped_charger_ids
        max_power_draw_per_group = self.station.grouped_charger_max_capacities
        assert len(max_power_draw_per_group) == len(idx_per_group)
        for i, charger_group in enumerate(idx_per_group):
            currents = currents.at[charger_group].set(
                actions[charger_group] * max_power_draw_per_group[i]
            )
        currents = jnp.minimum(currents, state.chargers_state.car_max_charge_rate) # max_charge_rate is 0 when no car is connected

        if self.renormalize_currents:
            currents = self.station.normalize_currents_recursive(currents, state.chargers_state)
        # bl = self.charger_topology.bottom_level
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

        new_charger_state = replace(
            state.chargers_state,
            charger_currrent_now=currents
        )

        # TODO: Renormalize power levels per group? (this now only occurs on the lowest level of chargers)
        # NOTE: this is probably already a fair assumption though.

        return replace(
            state,
            chargers=new_charger_state
        )
    
    def update_charging_levels(self, state: EnvState) -> EnvState:
        charge_past_timestep = state.chargers.charged_past_timestep(self.minutes_per_timestep) #state.chargers.charger_rate_current / 60 * self.minutes_per_timestep
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
    
    def process_buy_and_sell_electricity(self, state: EnvState) -> EnvState:
        total_power_draw_incl_losses = self.charger_topology.total_draw_group(state.chargers)
        total_power_draw_incl_losses = total_power_draw_incl_losses / 60 * self.minutes_per_timestep
        elec_price = self.elec_price_data(state.timestep)
        total_price_paid = total_power_draw_incl_losses.sum() * elec_price

        charged_past_timestep = state.chargers.charged_past_timestep(self.minutes_per_timestep)
        total_power_sold = jnp.sum(charged_past_timestep)
        total_price_received = total_power_sold * self.elec_sell_price

        return replace(
            state,
            profit=state.profit + (total_price_received - total_price_paid)
        )

    
    def update_time_and_clear_cars(self, state: EnvState) -> EnvState:

        car_waiting_times = state.chargers.time_till_leave - self.minutes_per_timestep
        car_waiting_times = jnp.maximum(car_waiting_times, 0)
        car_leaving = jnp.logical_and(car_waiting_times == 0, state.chargers.car_connected)
        cars_leaving_satisfied = jnp.logical_and(
            car_leaving, 
            state.chargers.battery_level >= 0.5
        ).sum()
        cars_leaving_unsatisfied = jnp.logical_and(
            car_leaving,
            state.chargers.battery_level < 0.5
        ).sum()
        car_connected = (car_waiting_times * state.chargers.car_connected).astype(bool)
        chargers = replace(
            state.chargers,
            time_till_leave=car_waiting_times,
            car_connected=car_connected,
        )
        return replace(state, 
            chargers=chargers,
            cars_leaving_satisfied=state.cars_leaving_satisfied + cars_leaving_satisfied,
            cars_leaving_unsatisfied=state.cars_leaving_unsatisfied + cars_leaving_unsatisfied
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
        # ev_arrival_means = [self.arrival_distributions[i]._loc for i in range(len(self.arrival_distributions))]
        # ev_arrival_stds = [self.arrival_distributions[i]._scale for i in range(len(self.arrival_distributions))]
        ev_arrival_means = jnp.array(self.ev_arrival_data_means)
        ev_arrival_stds = jnp.array(self.ev_arrival_data_stds)
        new_cars_amount = jax.random.normal(key, ()) * ev_arrival_stds[state.timestep] + ev_arrival_means[state.timestep]
        new_cars_amount = jnp.maximum(new_cars_amount, 0)


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

    def get_observation(self, state: EnvState) -> chex.Array:
        # next_price = self.elec_price_data(state.timestep + 1)
        # next_next_price = self.elec_price_data(state.timestep + 2)
        # next_price_grad = next_next_price - next_price
        # next_margin = self.elec_sell_price - self.elec_price_data(state.timestep)
        return jnp.array([0.0])

    # def get_action_masks(self, state: EnvState) -> chex.Array:
    #     raise NotImplementedError()

    def get_reward(self, old_state: EnvState, new_state: EnvState) -> chex.Array:
        return new_state.profit - old_state.profit
        return -(new_state.cars_leaving_unsatisfied - old_state.cars_leaving_unsatisfied)
        return new_state.cars_leaving_satisfied - old_state.cars_leaving_satisfied
        
    def get_terminated(self, state: EnvState) -> bool:
        return False
    
    def get_truncated(self, state: EnvState) -> bool:
        return state.timestep >= len(self.ev_arrival_data_means)
    
    def get_info(self, state: EnvState, actions) -> Dict[str, chex.Array]:
        return {
            "actions": actions
        }
    
    @property
    def observation_space(self):
        obs, _ = self.reset_env(jax.random.PRNGKey(0))
        return Box(-1, 1, obs.shape)
    
    @property
    def action_space(self):
        return MultiDiscrete(
            np.full(self.station.number_of_chargers, self.num_discretization_levels)
        )

if __name__ == "__main__":
    env = Chargax()
    key = jax.random.PRNGKey(0)
    env.reset(key)