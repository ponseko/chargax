import datetime
from dataclasses import asdict, replace
from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
from jaxnasium import TimeStep
from jaxtyping import Array, Float, PRNGKeyArray

from ._default_data_loaders import (
    build_default_grid_price_fn,
    build_default_scenario,
)
from ._station_layout import EVSE, ChargingStation, StationBattery


class EnvState(jym.EnvState):
    grid: ChargingStation
    day_of_year: int
    timestep: int = 0

    @property
    def is_workday(self) -> bool:
        """
        Determine if the current day is a workday (Monday to Friday).
        """
        offset = datetime.datetime(2024, 1, 1).weekday()  # 0 (change year if needed)
        day_of_week = (self.day_of_year + offset) % 7
        return day_of_week < 5

    # Reward variables:
    profit: float = 0.0
    uncharged_percentages: float = 0.0
    uncharged_kw: float = 0.0
    charged_overtime: int = 0
    charged_undertime: int = 0
    rejected_customers: int = 0
    left_customers: int = 0
    exceeded_capacity: float = 0.0
    total_charged_kw: float = 0.0
    total_discharged_kw: float = 0.0


class Chargax(jym.Environment):
    station: ChargingStation = eqx.field(static=True)
    elec_grid_buy_price: jnp.ndarray = eqx.field(converter=jnp.asarray)  # €/kWh
    elec_grid_sell_price: jnp.ndarray = eqx.field(converter=jnp.asarray)  # €/kWh
    elec_customer_sell_price: float = 0.75  # €/kWh

    get_num_cars_arriving: Callable[[PRNGKeyArray, EnvState], int] = None
    get_new_cars_arriving: Callable[[PRNGKeyArray, EnvState], EVSE] = None
    get_grid_buy_price: Callable[[EnvState], float] = None
    get_grid_sell_price: Callable[[EnvState], float] = None

    # reward alpha values
    capacity_exceeded_alpha: float = 0.0
    charged_satisfaction_alpha: float = 0.0
    time_satisfaction_alpha: float = 0.0
    rejected_customers_alpha: float = 0.0
    battery_degradation_alpha: float = 0.0
    beta: float = 0.0

    # Env options:
    num_discretization_levels: int = (
        10  # 10 would mean each charger can charge 10%, 20%, ... of its max rate
    )
    minutes_per_timestep: int = 5
    renormalize_currents: bool = True
    allow_discharging: bool = True
    price_hour_lookahead: int = 6

    full_info_dict: bool = False

    default_data_kwargs: Dict = eqx.field(static=True, default_factory=lambda: {})

    @property
    def max_episode_steps(self) -> int:
        return int(24 * 60 / self.minutes_per_timestep)  # Simulate one day

    def __post_init__(self):
        if self.get_num_cars_arriving is None or self.get_new_cars_arriving is None:
            get_num_cars, get_new_cars = build_default_scenario(
                self, **self.default_data_kwargs
            )
            if self.get_num_cars_arriving is None:
                self.__setattr__("get_num_cars_arriving", get_num_cars)
            if self.get_new_cars_arriving is None:
                self.__setattr__("get_new_cars_arriving", get_new_cars)

        dataset = self.default_data_kwargs.get("grid_price_dataset", "2023_NL")
        sell_price_margin = self.default_data_kwargs.get("grid_sell_margin", -0.03)
        if self.get_grid_buy_price is None:
            self.__setattr__(
                "get_grid_buy_price",
                build_default_grid_price_fn(self, dataset=dataset, offset=0),
            )

        if self.get_grid_sell_price is None:
            self.__setattr__(
                "get_grid_sell_price",
                build_default_grid_price_fn(
                    self, dataset=dataset, offset=sell_price_margin
                ),
            )

    def reset_env(self, key: PRNGKeyArray) -> Tuple[Dict[str, Array], EnvState]:

        state = EnvState(
            day_of_year=jax.random.randint(key, (), 0, 365),
            grid=self.station.zero_grid(),
        )
        observation = self.get_observation(state)
        return observation, state

    def step_env(
        self, rng: PRNGKeyArray, old_state: EnvState, actions: Dict[str, Array]
    ) -> Tuple[TimeStep, EnvState]:
        rng, key = jax.random.split(rng)
        new_state = old_state

        new_state = self.set_charging_currents(new_state, actions)

        charging_ports = new_state.grid.evses_flat
        batteries = new_state.grid.batteries_flat

        new_state, charging_ports = self.charge_cars(
            new_state, charging_ports, batteries
        )
        new_state, charging_ports = self.update_time_and_clear_cars(
            new_state, charging_ports
        )
        new_state, charging_ports = self.add_new_cars(key, new_state, charging_ports)

        # Set all chargers with no car connected to 0:
        charging_ports = jax.tree.map(
            lambda p: p * charging_ports.charger_is_car_connected, charging_ports
        )

        updated_grid = new_state.grid.update_evses_from_flat(
            charging_ports
        ).update_batteries_from_flat(batteries)

        new_state = new_state._replace(
            grid=updated_grid,
            timestep=old_state.timestep + 1,
        )

        timestep_object = jym.TimeStep(
            observation=self.get_observation(new_state),
            reward=self.get_reward(old_state, new_state),
            terminated=self.get_terminated(new_state),
            truncated=self.get_truncated(new_state),
            info=self.get_info(new_state, actions, old_state=old_state),
        )

        return timestep_object, new_state

    def set_charging_currents(self, state: EnvState, actions: Array) -> EnvState:
        """Set new currents and power levels based on actions"""

        def _set_action(item: EVSE | StationBattery, action: Array):
            action = action / self.num_discretization_levels

            if isinstance(item, EVSE):
                if self.allow_discharging:
                    action = action - 1

                # clip within car ranges and set current
                current = jnp.clip(
                    action * item.max_current,
                    -item.car_max_current_outtake if self.allow_discharging else 0,
                    item.car_max_current_intake,
                )  # car_max_current_intake is 0 when no car is connected
                return item.replace(charger_current_now=current)

            elif isinstance(item, StationBattery):
                # For the battery, directly set the output power and update the battery level
                action = action - 1
                desired_output_kw = action * item.max_rate_kw
                desired_output_kw_now = self.kw_to_kw_this_timestep(desired_output_kw)
                new_battery_level = jnp.clip(
                    item.battery_now + desired_output_kw_now, 0, item.capacity_kw
                )
                battery_change = new_battery_level - item.battery_now
                actual_output_kw = battery_change * (60 / self.minutes_per_timestep)
                return item.replace(
                    battery_now=new_battery_level, output_now_kw=actual_output_kw
                )

            raise ValueError("Invalid item type")

        updated_grid: ChargingStation = jax.tree.map(
            _set_action,
            state.grid,
            actions,
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery)),
        )
        if self.renormalize_currents:
            updated_grid = updated_grid.distribute()

        exceeded_capacity = updated_grid.exceeded_power_all_children

        return replace(
            state,
            grid=updated_grid,
            exceeded_capacity=state.exceeded_capacity + exceeded_capacity,
        )

    def charge_cars(
        self, state: EnvState, charging_ports: EVSE, batteries: StationBattery
    ) -> tuple[EnvState, EVSE]:

        charging_now = self.kw_to_kw_this_timestep(charging_ports.power_output)
        previous_battery = charging_ports.car_battery_now_kw
        new_battery = (previous_battery + charging_now).clip(
            charging_ports.car_arrival_battery_kw,  # can't discharge under arrival battery
            charging_ports.car_battery_capacity_kw,
        )
        real_charged_this_timestep = new_battery - previous_battery

        # Calculate customer revenue (EVSEs only)
        energy_sold = jnp.maximum(
            jnp.maximum(real_charged_this_timestep, 0.0)
            - charging_ports.car_discharged_this_session_kw,
            0.0,
        ).sum()
        revenue = energy_sold * self.elec_customer_sell_price
        discharged_this_session = (
            charging_ports.car_discharged_this_session_kw + -real_charged_this_timestep
        ).clip(0)

        grid_draw_evses = jnp.where(
            real_charged_this_timestep >= 0,
            real_charged_this_timestep / charging_ports.cumulative_efficiency,
            real_charged_this_timestep * charging_ports.cumulative_efficiency,
        )
        batteries_output_now_kw = self.kw_to_kw_this_timestep(batteries.output_now_kw)
        grid_draw_batteries = jnp.where(
            batteries_output_now_kw >= 0,
            batteries_output_now_kw / batteries.cumulative_efficiency,
            batteries_output_now_kw * batteries.cumulative_efficiency,
        )
        total_grid_draw = grid_draw_evses.sum() + grid_draw_batteries.sum()
        elec_price = jax.lax.select(
            total_grid_draw >= 0,
            self.get_grid_buy_price(state),
            self.get_grid_sell_price(state),
        )
        profit = state.profit + revenue - total_grid_draw * elec_price
        charging_ports = charging_ports.replace(
            car_discharged_this_session_kw=discharged_this_session,
            car_battery_now_kw=new_battery,
        )
        total_charged = jnp.maximum(real_charged_this_timestep, 0.0).sum()
        total_discharged = jnp.maximum(-real_charged_this_timestep, 0.0).sum()

        return replace(
            state,
            profit=profit,
            total_charged_kw=total_charged,
            total_discharged_kw=total_discharged,
        ), charging_ports

    def update_time_and_clear_cars(
        self, state: EnvState, ports: EVSE
    ) -> tuple[EnvState, EVSE]:
        new_time_till_leave = ports.car_time_till_leave - self.minutes_per_timestep
        new_time_waited = ports.car_time_waited + self.minutes_per_timestep
        is_leaving_time_sensitive = new_time_till_leave <= 0
        is_leaving_charge_sensitive = ports.car_battery_desired_remaining <= 0
        is_leaving = (is_leaving_charge_sensitive * ports.charge_sensitive) + (
            is_leaving_time_sensitive * ~ports.charge_sensitive
        )
        is_leaving = is_leaving * ports.charger_is_car_connected

        uncharged_percentages = (
            is_leaving * jnp.maximum(ports.car_battery_desired_remaining, 0)
        ).sum()
        uncharged_kw = (
            is_leaving * jnp.maximum(0, ports.car_battery_desired_remaining_kw)
        ).sum()
        charged_overtime = (
            jnp.abs(is_leaving * jnp.minimum(0, ports.car_time_till_leave))
            .sum()
            .astype(int)
        )  # Use previous time till leave to calculate overtime
        charged_undertime = (
            (is_leaving * jnp.maximum(0, new_time_till_leave)).sum().astype(int)
        )
        cars_leaving = is_leaving.sum()

        ports = ports.replace(
            car_time_till_leave=new_time_till_leave.astype(int),
            car_time_waited=new_time_waited,
            charger_is_car_connected=ports.charger_is_car_connected * ~is_leaving,
        )

        state = state._replace(
            uncharged_percentages=state.uncharged_percentages + uncharged_percentages,
            uncharged_kw=state.uncharged_kw + uncharged_kw,
            charged_overtime=state.charged_overtime + charged_overtime,
            charged_undertime=state.charged_undertime + charged_undertime,
            left_customers=state.left_customers + cars_leaving.sum(),
        )

        return state, ports

    def add_new_cars(
        self, key: PRNGKeyArray, state: EnvState, ports: EVSE
    ) -> tuple[EnvState, EVSE]:
        key1, key2 = jax.random.split(key)

        new_cars_amount = self.get_num_cars_arriving(key1, state)

        # Generate new chargers and put the car_connected to False when:
        # 1. The index of the charger is already connected to a car
        # 2. There are less incoming cars than chargers
        not_connected_chargers = jnp.logical_not(ports.charger_is_car_connected)
        sort_order = jnp.argsort(not_connected_chargers, descending=True)
        required_chargers = jnp.arange(self.station.num_chargers) < new_cars_amount
        required_chargers_in_order = (
            jnp.zeros_like(required_chargers).at[sort_order].set(required_chargers)
        )
        arrival_of_new_car_positions = (
            required_chargers_in_order * not_connected_chargers
        )  # adjust for overflow
        incoming_chargers = self.get_new_cars_arriving(key2, state)
        incoming_chargers = incoming_chargers.replace(
            charger_is_car_connected=arrival_of_new_car_positions,
        )
        # Merge the incoming chargers with the current chargers
        merged_chargers = jax.tree.map(
            lambda new, curr: jax.lax.select(arrival_of_new_car_positions, new, curr),
            incoming_chargers,
            ports,
        )

        rejected_customers = jnp.maximum(
            new_cars_amount - not_connected_chargers.sum(), 0
        ).astype(jnp.int32)

        state = state._replace(
            rejected_customers=state.rejected_customers + rejected_customers
        )

        return state, merged_chargers

    def get_observation(self, state: EnvState) -> Array:

        observations = {
            "evses": state.grid.evses,
            "batteries": state.grid.batteries,
        }

        # Get future prices
        timesteps_per_hour = 60 // self.minutes_per_timestep
        hour_offsets = jnp.arange(self.price_hour_lookahead) * timesteps_per_hour
        future_timesteps = state.timestep + hour_offsets
        future_prices = jax.vmap(
            lambda t: self.get_grid_buy_price(state._replace(timestep=t))
        )(future_timesteps)
        future_sell_prices = jax.vmap(
            lambda t: self.get_grid_sell_price(state._replace(timestep=t))
        )(future_timesteps)

        # Calculate price differences for all lookahead periods
        price_diffs_buy = future_prices[1:] - future_prices[0]  # all diffs from now
        price_diffs_sell = future_sell_prices[1:] - future_sell_prices[0]  # ""

        observations.update(
            {
                "future_buy_prices": future_prices,
                "future_sell_prices": future_sell_prices,
                "future_price_diffs_buy": price_diffs_buy,
                "future_price_diffs_sell": price_diffs_sell,
                "current_timestep": state.timestep,
                "current_day_of_year": state.day_of_year,
                "is_workday": state.is_workday,
            }
        )

        return observations

    def get_reward(self, old_state: EnvState, new_state: EnvState) -> Array:
        profit_delta = new_state.profit - old_state.profit

        # uncharged_delta = new_state.uncharged_percentages - old_state.uncharged_percentages
        uncharged_delta = new_state.uncharged_kw - old_state.uncharged_kw
        charged_overtime_delta = new_state.charged_overtime - old_state.charged_overtime
        charged_undertime_delta = (
            new_state.charged_undertime - old_state.charged_undertime
        )
        rejected_customers_delta = (
            new_state.rejected_customers - old_state.rejected_customers
        )
        exceeded_capacity_delta = (
            new_state.exceeded_capacity - old_state.exceeded_capacity
        )
        battery_degredation_delta = (
            new_state.total_discharged_kw - old_state.total_discharged_kw
        )  # use discharged kw as proxy for degredation

        return profit_delta - (
            self.charged_satisfaction_alpha * uncharged_delta
            + self.time_satisfaction_alpha
            * (charged_overtime_delta - (self.beta * charged_undertime_delta))
            + self.rejected_customers_alpha * rejected_customers_delta
            + self.capacity_exceeded_alpha * exceeded_capacity_delta
            + self.battery_degradation_alpha * battery_degredation_delta
        )

    def get_terminated(self, state: EnvState) -> bool:
        return False

    def get_truncated(self, state: EnvState) -> bool:
        return state.timestep >= self.max_episode_steps

    def get_info(
        self, state: EnvState, actions, old_state: EnvState = None
    ) -> Dict[str, Array]:
        if not self.full_info_dict:
            return {
                "logging_data": {
                    "profit": state.profit,
                    "exceeded_capacity": state.exceeded_capacity,
                    "total_charged_kw": state.total_charged_kw,
                    "total_discharged_kw": state.total_discharged_kw,
                    "rejected_customers": state.rejected_customers,
                    "uncharged_percentages": state.uncharged_percentages,
                    "uncharged_kw": state.uncharged_kw,
                    "charged_overtime": state.charged_overtime,
                    "charged_undertime": state.charged_undertime,
                    # "battery_level": state.grid.battery_now,
                    # "battery_percentage": state.battery_state.battery_percentage,
                }
            }
        return {
            "actions": actions,
            **asdict(state),
            # "car_battery_percentage": state.chargers_state.car_battery_percentage,
            # "car_battery_desired_remaining": state.chargers_state.car_battery_desired_remaining,
            # "charger_output_now_kw": state.chargers_state.charger_output_now_kw,
            # "charger_throughput_now_kw": state.chargers_state.charger_throughput_now_kw,
            # "car_max_current_intake": state.chargers_state.car_max_current_intake,
            # "car_max_current_outtake": state.chargers_state.car_max_current_outtake,
            "reward": self.get_reward(old_state, state),
        }

    def kw_to_kw_this_timestep(self, kw_drawn: Float[Array, "..."]) -> Array:
        return kw_drawn / 60 * self.minutes_per_timestep

    @property
    def observation_space(self):
        obs, _ = self.reset_env(jax.random.PRNGKey(0))
        return jax.tree.map(
            lambda v: jym.Box(-jnp.inf, jnp.inf, getattr(v, "shape", ())), obs
        )

    @property
    def action_space(self) -> jym.Space:
        """
        Define the action space of the environment.
        """
        num_actions_per_charger = self.num_discretization_levels
        if self.allow_discharging:
            num_actions_per_charger *= 2
        num_actions_per_battery = self.num_discretization_levels * 2

        def _create_action_space(item: EVSE | StationBattery) -> jym.Space:
            if isinstance(item, EVSE):
                return jym.MultiDiscrete(
                    np.full(item.num_chargers, num_actions_per_charger)
                )
            elif isinstance(item, StationBattery):
                return jym.Discrete(num_actions_per_battery)

        return jax.tree.map(
            _create_action_space,
            self.station,
            is_leaf=lambda x: isinstance(x, (EVSE, StationBattery)),
        )
