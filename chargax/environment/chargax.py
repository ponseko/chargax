from typing import Tuple, Dict, List, Callable, Type, Union
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx 
from dataclasses import replace, asdict
import chex

from chargax import JaxBaseEnv, TimeStep, EnvState, ChargersState, MultiDiscrete, Box, ChargingStation

# disable jit
# jax.config.update("jax_disable_jit", True)


def pre_sample_data(data: np.ndarray, num_samples = 10000) -> np.ndarray:
    data = np.asarray(data)
    data = np.random.poisson(data, (num_samples, len(data)))
    return jnp.array(data)

class Chargax(JaxBaseEnv):
    
    # charger_topology: ChargerGroup
    
    ev_arrival_means_workdays: np.ndarray = eqx.field(converter=pre_sample_data)
    ev_arrival_means_non_workdays: np.ndarray = eqx.field(converter=pre_sample_data)
    elec_grid_buy_price: Callable = lambda x: jnp.sin(x) + 1.25 
    elec_grid_sell_diff: Callable = 0.10 # €/kWh
    elec_customer_sell_price: float = 0.75 # €/kWh

    station: ChargingStation = ChargingStation()

    num_discretization_levels: int = 10 # 10 would mean each charger can charge 10%, 20%, ... of its max rate
    minutes_per_timestep: int = 5
    renormalize_currents: bool = True
    include_battery: bool = True
    allow_discharging: bool = True
    
    def __post_init__(self):
        # self.__setattr__("some_property", some_property_value)
        pass

    def __check_init__(self):
        # assert anything ...
        pass 

    def reset_env(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        state = EnvState(
            day_of_year=jax.random.randint(key, (), 0, 365),
            chargers_state=ChargersState(self.station, sample_method="empty"),
        )
        observation = self.get_observation(state)
        return observation, state
    
    def step_env(self, rng: chex.PRNGKey, old_state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[TimeStep, EnvState]:

        rng, key = jax.random.split(rng)
        new_state = old_state

        actions = self.preprocess_actions(actions)

        new_state = self.set_currents_and_update_charge_levels(new_state, actions)
        new_state = self.process_grid_transactions(old_state, new_state)
        new_state = self.update_time_and_clear_cars(new_state)
        new_state = self.add_new_cars(new_state, key)

        # Set all variables of left cars to 0:
        charger_state_vars, cs_structure = jax.tree.flatten(new_state.chargers_state)
        charger_state_vars = jax.tree.map(
            lambda x: x * new_state.chargers_state.charger_is_car_connected,
            charger_state_vars
        )
        charger_state = jax.tree.unflatten(cs_structure, charger_state_vars)


        new_state = replace(
            new_state,
            chargers_state=charger_state,
            timestep=old_state.timestep + 1 # advance time while we are at it
        )

        timestep_object = TimeStep(
            observation=self.get_observation(new_state),
            reward=self.get_reward(old_state, new_state),
            terminated=self.get_terminated(new_state),
            truncated=self.get_truncated(new_state),
            info=self.get_info(new_state, actions)
        )
    
        return timestep_object, new_state
    
    def preprocess_actions(self, actions: chex.Array) -> chex.Array:
        actions = actions / self.num_discretization_levels
        if self.allow_discharging:
            actions = actions - 1
        return actions
    
    def set_currents_and_update_charge_levels(self, state: EnvState, actions: chex.Array) -> EnvState:

        idx_per_group = self.station.charger_ids_per_evse
        max_current_per_evse = [evse.current_max for evse in self.station.evses]
        assert len(max_current_per_evse) == len(idx_per_group)

        currents = jnp.zeros(self.station.num_chargers)
        for i, charger_group in enumerate(idx_per_group):
            currents = currents.at[charger_group].set(
                actions[charger_group] * max_current_per_evse[i]
            )

        currents = jnp.clip(
            currents, 
            -state.chargers_state.car_max_current_intake,
            state.chargers_state.car_max_current_intake
        )  # car_max_current_intake is 0 when no car is connected
        
        charger_state = replace(
            state.chargers_state,
            charger_current_now=currents
        )

        if self.renormalize_currents:
            # NOTE: I am not sure if looping over the splitters backwards is the correct order
            for splitter in self.station.splitters[::-1]:
                charger_state = splitter.normalize_currents(charger_state)
            charger_state = self.station.root.normalize_currents(charger_state) 

        ### Update Car Battery Levels
        charged_this_timestep = self.kw_to_kw_this_timestep(
            charger_state.charger_output_now_kw
        )
        new_car_batteries = jnp.clip(
            state.chargers_state.car_battery_now + charged_this_timestep,
            0,
            state.chargers_state.car_battery_capacity
        )
        charger_state = replace(
            charger_state,
            car_battery_now=new_car_batteries
        )

        ### Update Station Battery Levels
        new_battery_state = state.battery_state
        if self.include_battery:
            battery_action = actions[-1]
            battery_charge_or_discharge = battery_action * state.battery_state.max_rate_kw
            battery_charge_or_discharge = self.kw_to_kw_this_timestep(battery_charge_or_discharge)
            new_battery_charge = jnp.clip(
                state.battery_state.battery_now + battery_charge_or_discharge,
                0,
                state.battery_state.capacity_kw
            )
            new_battery_state = replace(
                state.battery_state,
                battery_now=new_battery_charge
            )

        return replace(
            state,
            chargers_state=charger_state,
            battery_state=new_battery_state
        )
    
    def process_grid_transactions(self, old_state: EnvState, new_state: EnvState) -> EnvState:

        charged_this_timestep = (
            new_state.chargers_state.car_battery_now - old_state.chargers_state.car_battery_now 
        )

        energy_exchanged_w_customers = charged_this_timestep.sum()
        customer_revenue = energy_exchanged_w_customers * self.elec_sell_price # sell/buy price is the same for customers

        grid_energy_transported = jnp.where( # adjust for efficiency
            # When charging, add losses. When discharging, subtract losses
            charged_this_timestep >= 0,
            charged_this_timestep * self.station.root.efficiency_per_charger,
            charged_this_timestep / self.station.root.efficiency_per_charger
        ).sum()
        if self.include_battery:
            grid_energy_transported += (
                new_state.battery_state.battery_now - old_state.battery_state.battery_now
            )

        elec_price = jax.lax.select(
            grid_energy_transported >= 0,
            self.elec_price_data(old_state.timestep), # Buying from grid
            self.elec_grid_sell_price_data(old_state.timestep) # Selling to grid
        )
        energy_cost = grid_energy_transported * elec_price # $/kWh -- negative when selling

        profit = new_state.profit + customer_revenue - energy_cost

        return replace(
            new_state,
            profit=profit
        )
    
    def update_time_and_clear_cars(self, state: EnvState) -> EnvState:

        car_waiting_times = state.chargers_state.car_time_till_leave - self.minutes_per_timestep
        # car_waiting_times = jnp.maximum(car_waiting_times, 0)
        
        time_sensitive_leaving = jnp.logical_and(
            car_waiting_times <= 0, 
            ~state.chargers_state.charge_sensitive
        )
        charge_sensitive_leaving = jnp.logical_and(
            state.chargers_state.car_battery_desired_remaining <= 0,
            state.chargers_state.charge_sensitive
        )
        cars_leaving = jnp.logical_or(
            time_sensitive_leaving, charge_sensitive_leaving
        ) * state.chargers_state.charger_is_car_connected

        uncharged_percentages = (cars_leaving * state.chargers_state.car_battery_desired_remaining).sum()
        charged_overtime = (cars_leaving * jnp.maximum(car_waiting_times, 0)).sum()
        charged_undertime = (cars_leaving * jnp.minimum(car_waiting_times, 0)).sum()
        
        car_connected = state.chargers_state.charger_is_car_connected * ~cars_leaving
        return replace(
            state, 
            chargers_state=replace(
                state.chargers_state,
                car_time_till_leave=car_waiting_times,
                charger_is_car_connected=car_connected,
            ),
            uncharged_percentages=state.uncharged_percentages + uncharged_percentages,
            charged_overtime=state.charged_overtime + charged_overtime,
            charged_undertime=state.charged_undertime + charged_undertime,
        )
    
    def add_new_cars(self, state: EnvState, key: chex.PRNGKey) -> EnvState:

        # new_cars_amount = jax.random.poisson(key, jnp.array(self.ev_arrival_data_means)[state.timestep])
        # new_cars_amount = jnp.maximum(new_cars_amount, 0)
        new_cars_amount = self.ev_arrival_means_workdays[state.day_of_year][state.timestep]

        # Generate new chargers and put the car_connected to False when:
        # 1. The index of the charger is already connected to a car
        # 2. There are less incoming cars than chargers
        not_connected_chargers = jnp.logical_not(state.chargers_state.charger_is_car_connected)
        sort_order = jnp.argsort(not_connected_chargers, descending=True)
        required_chargers = jnp.arange(self.station.num_chargers) < new_cars_amount
        required_chargers_in_order = jnp.zeros_like(required_chargers).at[sort_order].set(required_chargers)
        incoming_chargers = ChargersState(self.station, sample_method="random", key=key)
        incoming_chargers = replace(
            incoming_chargers, 
            charger_is_car_connected=required_chargers_in_order,
        )
        # Merge the incoming chargers with the current chargers
        merged_chargers = jax.tree.map(
            lambda new, curr: jax.lax.select(
                required_chargers_in_order, new, curr
            ), incoming_chargers, state.chargers_state
        )

        rejected_customers = jnp.maximum(new_cars_amount - not_connected_chargers.sum(), 0).astype(jnp.int32)

        return replace(
            state, 
            chargers_state=merged_chargers,
            rejected_customers=state.rejected_customers + rejected_customers
        )

    def get_observation(self, state: EnvState) -> chex.Array:

        charger_state = state.chargers_state
        battery_state = state.battery_state

        observations = jnp.concatenate([
            charger_state.car_time_till_leave,
            charger_state.car_battery_now,
            charger_state.car_battery_capacity,
            # charger_state.car_desired_battery_percentage,
            charger_state.charge_sensitive,
            charger_state.car_battery_percentage,
            charger_state.car_battery_desired_remaining,
            charger_state.car_max_current_intake,

            charger_state.car_ac_absolute_max_charge_rate_kw,
            charger_state.car_ac_optimal_charge_threshold,
            charger_state.car_dc_absolute_max_charge_rate_kw,
            charger_state.car_dc_optimal_charge_threshold,
            charger_state.charger_output_now_kw,
        ])
        if self.include_battery:
            observations = jnp.concatenate([
                observations,
                jnp.array([battery_state.battery_now, battery_state.battery_percentage, battery_state.max_rate_kw])
            ])


        

        return observations

    # def get_action_masks(self, state: EnvState) -> chex.Array:
    #     raise NotImplementedError()

    def get_reward(self, old_state: EnvState, new_state: EnvState) -> chex.Array:
        profit_delta = new_state.profit - old_state.profit
        
        uncharged_delta = new_state.uncharged_percentages - old_state.uncharged_percentages
        charged_overtime_delta = new_state.charged_overtime - old_state.charged_overtime
        charged_undertime_delta = new_state.charged_undertime - old_state.charged_undertime
        rejected_customers_delta = new_state.rejected_customers - old_state.rejected_customers

        return profit_delta - (
              0.1 * uncharged_delta
            + 0.1 * charged_overtime_delta
            + 0.1 * charged_undertime_delta
            + 0.1 * rejected_customers_delta
        )
        
    def get_terminated(self, state: EnvState) -> bool:
        return False
    
    def get_truncated(self, state: EnvState) -> bool:
        return state.timestep >= len(self.ev_arrival_means_workdays[0])
    
    def get_info(self, state: EnvState, actions) -> Dict[str, chex.Array]:
        return {
            "actions": actions
        }
    
    def kw_to_kw_this_timestep(self, kw_drawn: Union[float, np.ndarray[float]]) -> chex.Array:
        return kw_drawn / 60 * self.minutes_per_timestep
    
    @property
    def observation_space(self):
        obs, _ = self.reset_env(jax.random.PRNGKey(0))
        return Box(-1, 1, obs.shape)
    
    @property
    def action_space(self):
        num_action_objects = self.station.num_chargers
        num_actions_per_object = self.num_discretization_levels
        if self.include_battery:
            num_action_objects += 1
        if self.allow_discharging:
            num_actions_per_object *= 2
        
        return MultiDiscrete(np.full(num_action_objects, num_actions_per_object))

if __name__ == "__main__":
    env = Chargax()
    key = jax.random.PRNGKey(0)
    env.reset(key)