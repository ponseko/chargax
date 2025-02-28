from typing import Tuple, Dict, List, Callable, Type, Union, Literal
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx 
from dataclasses import replace, asdict
import chex
import datetime

from chargax import JaxBaseEnv, TimeStep, EnvState, ChargersState, MultiDiscrete, Box, ChargingStation
from ._data_loaders import get_scenario, get_car_data

# disable jit
# jax.config.update("jax_disable_jit", True)


class Chargax(JaxBaseEnv):
    
    elec_grid_buy_price: jnp.ndarray = eqx.field(converter=jnp.asarray) # €/kWh
    elec_grid_sell_price: jnp.ndarray = eqx.field(converter=jnp.asarray) # €/kWh
    elec_customer_sell_price: float = 0.75 # €/kWh

    # Data:
    ev_arrival_means_workdays: jnp.ndarray = None
    ev_arrival_means_non_workdays: jnp.ndarray = None

    car_profiles: Literal["eu", "us", "world", "custom"] = eqx.field(converter=str.lower, default="eu")
    user_profiles: Literal["highway", "residential", "workplace", "shopping", "custom"] = eqx.field(converter=str.lower, default="shopping")
    arrival_frequency: Union[int, Literal["low", "medium", "high"]] = 100

    # Station:
    station: ChargingStation = ChargingStation()
    num_chargers: int = 16
    num_chargers_per_group: int = 2
    num_dc_groups: int = 5

    # reward alpha values
    capacity_exceeded_alpha: float = 0.0
    charged_satisfaction_alpha: float = 0.0
    time_satisfaction_alpha: float = 0.0
    rejected_customers_alpha: float = 0.0
    battery_degredation_alpha: float = 0.0
    beta: float = 0.0

    # Env options:
    num_discretization_levels: int = 10 # 10 would mean each charger can charge 10%, 20%, ... of its max rate
    minutes_per_timestep: int = 5
    renormalize_currents: bool = True
    include_battery: bool = True
    allow_discharging: bool = True

    full_info_dict: bool = False
    
    def __post_init__(self):
        
        def pre_sample_data(data: np.ndarray, num_samples = 10000) -> jnp.ndarray:
            data = np.asarray(data)
            data = np.random.poisson(data, (num_samples, len(data)))
            return jnp.array(data)
        
        if self.arrival_frequency in ["low", "medium", "high"]:
            if self.arrival_frequency == "low":
                arrival_frequency = 50
            elif self.arrival_frequency == "medium":
                arrival_frequency = 100
            elif self.arrival_frequency == "high":
                arrival_frequency = 250
        else:
            arrival_frequency = self.arrival_frequency
        
        arrival_data_workdays, arrival_data_weekends, _, _ = get_scenario(self.user_profiles, average_cars_per_day=arrival_frequency, minutes_per_timestep=self.minutes_per_timestep)
        self.__setattr__("ev_arrival_means_workdays", pre_sample_data(arrival_data_workdays, num_samples=(10000 // 7) * 5))
        self.__setattr__("ev_arrival_means_non_workdays", pre_sample_data(arrival_data_weekends, num_samples=(10000 // 7) * 2))

        station = ChargingStation(
            num_chargers=self.num_chargers, 
            num_chargers_per_group=self.num_chargers_per_group, 
            num_dc_groups=self.num_dc_groups
        )
        self.__setattr__("station", station)

        # self.__setattr__("some_property", some_property_value)

    def __check_init__(self):
        # assert anything ...
        pass 

    def reset_env(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        day_of_year = jax.random.randint(key, (), 0, 365) # 0-364 (sort of exploring starts)
        
        # we assume the year is 2024 for now ... check if the day is a workday or weekendday
        def get_is_workday(day_of_year: int, offset: int = datetime.datetime(2024, 1, 1).weekday()) -> bool:
            """
            offset=2 would make day 0 a Wednesday (i.e., if Jan 1 is a Wed).
            """
            day_of_week = (day_of_year + offset) % 7
            return day_of_week < 5

        is_workday = get_is_workday(day_of_year)
        state = EnvState(
            day_of_year=day_of_year,
            chargers_state=ChargersState(self.station),
            is_workday=is_workday,
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

        # Set all variables of cars that have left to 0:
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
            info=self.get_info(new_state, actions, old_state=old_state)
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
            -state.chargers_state.car_max_current_outtake if self.allow_discharging else 0,
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
        tried_charged_this_timestep = self.kw_to_kw_this_timestep(
            charger_state.charger_output_now_kw
        )
        new_car_batteries = jnp.clip(
            state.chargers_state.car_battery_now_kw + tried_charged_this_timestep,
            state.chargers_state.car_arrival_battery_kw, # can't discharge under arrival battery
            state.chargers_state.car_battery_capacity_kw
        )
        actual_charged_this_timestep = new_car_batteries - state.chargers_state.car_battery_now_kw

        # update discharged
        discharged_this_session = jnp.maximum(
            state.chargers_state.car_discharged_this_session_kw + -actual_charged_this_timestep,
            0
        )

        charger_state = replace(
            charger_state,
            car_battery_now_kw=new_car_batteries,
            car_discharged_this_session_kw=discharged_this_session
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

        exceeded_capacity = 0.
        for splitter in self.station.splitters:
            exceeded_capacity += jnp.maximum(
                splitter.total_kw_throughput(charger_state) - splitter.group_capacity_max_kw,
                0
            )

        charging_cars_now = jnp.maximum(0, charger_state.charger_output_now_kw)
        discharging_cars_now = jnp.minimum(0, charger_state.charger_output_now_kw)
        charging_cars_now_kw = self.kw_to_kw_this_timestep(charging_cars_now).sum()
        discharging_cars_now_kw = jnp.abs(self.kw_to_kw_this_timestep(discharging_cars_now).sum())

        battery_change = state.battery_state.battery_now - new_battery_state.battery_now
        charging_battery_now = jnp.maximum(0, battery_change)
        discharging_battery_now = jnp.abs(jnp.minimum(0, battery_change))

        total_charged = charging_cars_now_kw + charging_battery_now
        total_discharged = discharging_cars_now_kw + discharging_battery_now

        return replace(
            state,
            chargers_state=charger_state,
            battery_state=new_battery_state,
            exceeded_capacity=exceeded_capacity,
            total_charged_kw=state.total_charged_kw + total_charged,
            total_discharged_kw=state.total_discharged_kw + total_discharged
        )
    
    def process_grid_transactions(self, old_state: EnvState, new_state: EnvState) -> EnvState:

        charged_this_timestep = (
            new_state.chargers_state.car_battery_now_kw - old_state.chargers_state.car_battery_now_kw 
        )

        energy_sold_to_customers = jnp.maximum(charged_this_timestep, 0)
        # some energy should be free because it was discharged earlier
        discharged_earlier = old_state.chargers_state.car_discharged_this_session_kw
        energy_sold_to_customers = jnp.maximum(
            energy_sold_to_customers - discharged_earlier,
            0
        )
        customer_revenue = energy_sold_to_customers.sum() * self.elec_customer_sell_price

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
            self.elec_grid_buy_price[new_state.day_of_year][new_state.timestep], # Buying from grid
            self.elec_grid_sell_price[new_state.day_of_year][new_state.timestep] # Selling to grid
        )
        energy_cost = grid_energy_transported * elec_price # $/kWh -- negative when selling

        profit = new_state.profit + customer_revenue - energy_cost

        return replace(
            new_state,
            profit=profit
        )
    
    def update_time_and_clear_cars(self, state: EnvState) -> EnvState:

        car_time_till_leave = state.chargers_state.car_time_till_leave - self.minutes_per_timestep
        
        time_sensitive_leaving = jnp.logical_and(
            car_time_till_leave <= 0, 
            ~state.chargers_state.charge_sensitive
        )
        charge_sensitive_leaving = jnp.logical_and(
            state.chargers_state.car_battery_desired_remaining <= 0,
            state.chargers_state.charge_sensitive
        )
        cars_leaving = jnp.logical_or(
            time_sensitive_leaving, charge_sensitive_leaving
        ) * state.chargers_state.charger_is_car_connected

        uncharged_percentages = (cars_leaving * jnp.maximum(0, state.chargers_state.car_battery_desired_remaining)).sum()
        uncharged_kw = (cars_leaving * jnp.maximum(0, state.chargers_state.car_battery_desired_remaining_kw)).sum()
        
        charged_overtime = jnp.abs(
            cars_leaving * jnp.minimum(
                car_time_till_leave + self.minutes_per_timestep, # add back the current timestep 
                0
            )).sum()
        charged_undertime = (cars_leaving * jnp.maximum(car_time_till_leave, 0)).sum()

        # # Compensate cars that leave with less battery than they arrived with
        # # NOTE: could also block this, but lets leave the agent flexible
        # kw_to_compensate = jnp.where(
        #     jnp.logical_and(
        #         state.chargers_state.car_battery_now_kw < state.chargers_state.car_arrival_battery_kw,
        #         state.chargers_state.car_battery_desired_remaining > 0
        #     ),
        #     state.chargers_state.car_arrival_battery_kw - state.chargers_state.car_battery_now_kw,
        #     0
        # ) * cars_leaving
        # currency_to_compensate = kw_to_compensate.sum() * self.elec_customer_sell_price
        # profit = state.profit - currency_to_compensate
        
        car_connected = state.chargers_state.charger_is_car_connected * ~cars_leaving

        return replace(
            state, 
            chargers_state=replace(
                state.chargers_state,
                car_time_till_leave=car_time_till_leave,
                charger_is_car_connected=car_connected,
            ),
            uncharged_percentages=state.uncharged_percentages + uncharged_percentages,
            uncharged_kw=state.uncharged_kw + uncharged_kw,
            charged_overtime=state.charged_overtime + charged_overtime,
            charged_undertime=state.charged_undertime + charged_undertime,
            left_customers=state.left_customers + cars_leaving.sum(),
            # profit=profit
        )
    
    def add_new_cars(self, state: EnvState, key: chex.PRNGKey) -> EnvState:

        key1, key2 = jax.random.split(key)

        # poission dists are pre-sampled -- but each is independent, so we can just 
        # draw any of the presampled batches at each timestep
        i = jax.random.randint(key1, (), 0, self.ev_arrival_means_workdays.shape[0])
        new_cars_amount = jax.lax.select(
            state.is_workday,
            self.ev_arrival_means_workdays[i][state.timestep],
            self.ev_arrival_means_non_workdays[i][state.timestep]
        )

        # Generate new chargers and put the car_connected to False when:
        # 1. The index of the charger is already connected to a car
        # 2. There are less incoming cars than chargers
        not_connected_chargers = jnp.logical_not(state.chargers_state.charger_is_car_connected)
        sort_order = jnp.argsort(not_connected_chargers, descending=True)
        required_chargers = jnp.arange(self.station.num_chargers) < new_cars_amount
        required_chargers_in_order = jnp.zeros_like(required_chargers).at[sort_order].set(required_chargers)
        arrival_of_new_car_positions = required_chargers_in_order * not_connected_chargers # adjust for overflow
        incoming_chargers = self.sample_cars(key2)
        incoming_chargers = replace(
            incoming_chargers, 
            charger_is_car_connected=arrival_of_new_car_positions,
        )
        # Merge the incoming chargers with the current chargers
        merged_chargers = jax.tree.map(
            lambda new, curr: jax.lax.select(
                arrival_of_new_car_positions, new, curr
            ), incoming_chargers, state.chargers_state
        )

        rejected_customers = jnp.maximum(new_cars_amount - not_connected_chargers.sum(), 0).astype(jnp.int32)

        return replace(
            state, 
            chargers_state=merged_chargers,
            rejected_customers=state.rejected_customers + rejected_customers
        )

    def sample_cars(self, key: chex.PRNGKey) -> ChargersState:
        """
        Returns a ChargersState based on the current scenario (user and car profiles)
        In the returned ChargersState, all connections are filled
        with is_car_connected to false. The required number of chargers should then 
        be connected and then merged with the current ChargersState.
        """ 
        chargers_state = ChargersState(self.station)
        if self.car_profiles in ["eu", "us", "world"]:
            car_data = jnp.array(get_car_data(self.car_profiles))
            probs = car_data[:, 0]
            cars = jax.random.categorical(key, probs, shape=(self.station.num_chargers,))
            tau_car_data = car_data[:, 1]
            capacity_car_data = car_data[:, 2]
            ac_max_rate_car_data = car_data[:, 3]
            dc_max_rate_car_data = car_data[:, 4]
            chargers_state = replace(
                chargers_state,
                car_ac_absolute_max_charge_rate_kw=ac_max_rate_car_data[cars],
                car_ac_optimal_charge_threshold=tau_car_data[cars],
                car_dc_absolute_max_charge_rate_kw=dc_max_rate_car_data[cars],
                car_dc_optimal_charge_threshold=tau_car_data[cars],
                car_battery_capacity_kw=capacity_car_data[cars]
            )

        if self.user_profiles in ["highway", "residential", "workplace", "shopping"]:
            _, _, connection_times, energy_demands = get_scenario(self.user_profiles)
            keys = jax.random.split(key, 3)
            connection_times_rnd = jax.random.randint(keys[0], (self.station.num_chargers,), 0, 101)
            energy_demands_rnd = jax.random.randint(keys[1], (self.station.num_chargers,), 0, 101)
            car_time_till_leave = connection_times[connection_times_rnd].astype(int)

            energy_demands = energy_demands[energy_demands_rnd]
            car_desired_battery_percentage = jax.random.uniform(keys[2], (self.station.num_chargers,), minval=0.8, maxval=0.95)
            car_desired_kw = chargers_state.car_battery_capacity_kw * car_desired_battery_percentage
            car_battery_now_kw = car_desired_kw - energy_demands
            car_battery_now_kw = jnp.clip(
                car_battery_now_kw,
                0.03 * chargers_state.car_battery_capacity_kw,
                chargers_state.car_battery_capacity_kw
            )

            
            # current_batteries = chargers_state.car_battery_capacity_kw - energy_demands
            # car_battery_now_kw = jnp.clip(
            #     current_batteries,
            #     0.03 * chargers_state.car_battery_capacity_kw,
            #     chargers_state.car_battery_capacity_kw
            # )

            # car_desired_battery_percentage = jax.random.uniform(keys[2], (self.station.num_chargers,), minval=0.8, maxval=0.95)
            # car_desired_battery_percentage = jnp.maximum(
            #     car_desired_battery_percentage,
            #     chargers_state.car_battery_percentage # users can't desire less than what they have
            # )
            if self.user_profiles == "highway":
                charge_sensitive = jax.random.bernoulli(keys[2], 0.9, shape=(self.station.num_chargers,))
            else:
                charge_sensitive = jax.random.bernoulli(keys[2], 0.1, shape=(self.station.num_chargers,))

            chargers_state = replace(
                chargers_state,
                car_time_till_leave=car_time_till_leave,
                car_battery_now_kw=car_battery_now_kw,
                car_desired_battery_percentage=car_desired_battery_percentage,
                charge_sensitive=charge_sensitive
            )

        return replace(
            chargers_state,
            car_arrival_battery_kw=chargers_state.car_battery_now_kw, # copy the initial battery percentage
        )

    def get_observation(self, state: EnvState) -> chex.Array:

        charger_state = state.chargers_state
        battery_state = state.battery_state

        observations = jnp.concatenate([
            charger_state.car_time_till_leave,
            charger_state.car_battery_now_kw,
            charger_state.car_battery_capacity_kw,
            # charger_state.car_desired_battery_percentage,
            charger_state.charge_sensitive,
            charger_state.car_battery_percentage,
            charger_state.car_battery_desired_remaining,
            charger_state.car_max_current_intake,
            charger_state.car_max_current_outtake,
            charger_state.car_discharged_this_session_kw,
            charger_state.car_arrival_battery_kw,
            state.chargers_state.car_battery_now_kw < state.chargers_state.car_arrival_battery_kw,
            state.chargers_state.car_arrival_battery_kw - state.chargers_state.car_battery_now_kw,

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

        timesteps_per_hour = 60 // self.minutes_per_timestep
        price_now = self.elec_grid_buy_price[state.day_of_year][state.timestep]
        price_plus_1 = self.elec_grid_buy_price[state.day_of_year][(state.timestep + timesteps_per_hour)]
        price_plus_2 = self.elec_grid_buy_price[state.day_of_year][(state.timestep + timesteps_per_hour * 2)]
        price_plus_3 = self.elec_grid_buy_price[state.day_of_year][(state.timestep + timesteps_per_hour * 3)]
        price_plus_4 = self.elec_grid_buy_price[state.day_of_year][(state.timestep + timesteps_per_hour * 4)]
        price_plus_5 = self.elec_grid_buy_price[state.day_of_year][(state.timestep + timesteps_per_hour * 5)]
        price_diff_next = price_plus_1 - price_now

        sell_price_now = self.elec_grid_sell_price[state.day_of_year][state.timestep]
        sell_price_plus_1 = self.elec_grid_sell_price[state.day_of_year][(state.timestep + timesteps_per_hour)]
        sell_price_plus_2 = self.elec_grid_sell_price[state.day_of_year][(state.timestep + timesteps_per_hour * 2)]
        sell_price_plus_3 = self.elec_grid_sell_price[state.day_of_year][(state.timestep + timesteps_per_hour * 3)]
        sell_price_plus_4 = self.elec_grid_sell_price[state.day_of_year][(state.timestep + timesteps_per_hour * 4)]
        sell_price_plus_5 = self.elec_grid_sell_price[state.day_of_year][(state.timestep + timesteps_per_hour * 5)]
        price_diff_next_sell = sell_price_plus_1 - sell_price_now


        observations = jnp.concatenate([
            observations,
            jnp.array([
                state.timestep, 
                state.day_of_year,
                state.is_workday,
                price_now, price_plus_1, price_plus_2, price_plus_3, price_plus_4, price_plus_5,
                sell_price_now, sell_price_plus_1, sell_price_plus_2, sell_price_plus_3, sell_price_plus_4, sell_price_plus_5,
                price_diff_next, price_diff_next_sell
            ])
        ])

        return observations

    # def get_action_masks(self, state: EnvState) -> chex.Array:
    #     raise NotImplementedError()

    def get_reward(self, old_state: EnvState, new_state: EnvState) -> chex.Array:
        profit_delta = new_state.profit - old_state.profit
        
        # uncharged_delta = new_state.uncharged_percentages - old_state.uncharged_percentages
        uncharged_delta = new_state.uncharged_kw - old_state.uncharged_kw
        charged_overtime_delta = new_state.charged_overtime - old_state.charged_overtime
        charged_undertime_delta = new_state.charged_undertime - old_state.charged_undertime
        rejected_customers_delta = new_state.rejected_customers - old_state.rejected_customers
        exceeded_capacity_delta = new_state.exceeded_capacity - old_state.exceeded_capacity
        battery_degredation_delta = new_state.total_discharged_kw - old_state.total_discharged_kw # use discharged kw as proxy for degredation

        return profit_delta - (
              self.charged_satisfaction_alpha * uncharged_delta
            + self.time_satisfaction_alpha * (charged_overtime_delta - (self.beta * charged_undertime_delta))
            + self.rejected_customers_alpha * rejected_customers_delta
            + self.capacity_exceeded_alpha * exceeded_capacity_delta
            + self.battery_degredation_alpha * battery_degredation_delta
        )
        
    def get_terminated(self, state: EnvState) -> bool:
        return False
    
    def get_truncated(self, state: EnvState) -> bool:
        return state.timestep >= self.episode_length
    
    def get_info(self, state: EnvState, actions, old_state: EnvState = None) -> Dict[str, chex.Array]:
        if not self.full_info_dict:
            return {
                "logging_data": {
                    "profit": state.profit,
                    "exceeded_capacity": state.exceeded_capacity,
                    "total_charged_kw": state.total_charged_kw,
                    "total_discharged_kw": state.total_discharged_kw,
                    "rejected_customers": state.rejected_customers,
                    "uncharged_percentages" : state.uncharged_percentages,
                    "uncharged_kw" : state.uncharged_kw,
                    "charged_overtime": state.charged_overtime,
                    "charged_undertime": state.charged_undertime,
                    "battery_level": state.battery_state.battery_now,
                    "battery_percentage": state.battery_state.battery_percentage,
                }
            }
        return {
            "actions": actions,
            **asdict(state),
            "car_battery_percentage": state.chargers_state.car_battery_percentage,
            "car_battery_desired_remaining": state.chargers_state.car_battery_desired_remaining,
            "charger_output_now_kw": state.chargers_state.charger_output_now_kw,
            "charger_throughput_now_kw": state.chargers_state.charger_throughput_now_kw,
            "car_max_current_intake": state.chargers_state.car_max_current_intake,
            "car_max_current_outtake": state.chargers_state.car_max_current_outtake,
            "reward": self.get_reward(old_state, state),
        }
    
    def kw_to_kw_this_timestep(self, kw_drawn: Union[float, np.ndarray[float]]) -> chex.Array:
        return kw_drawn / 60 * self.minutes_per_timestep
    
    @property
    def episode_length(self):
        return len(self.ev_arrival_means_workdays[0])
    
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