from __future__ import annotations

import csv
from importlib import resources as r
from typing import TYPE_CHECKING, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

if TYPE_CHECKING:
    from .chargax import EVSE, Chargax, EnvState

DATA_FOLDER = "data"


def _average_data(data, length):
    """
    Average over data to a desired length
    i.e. array([0, 5, 10]) -> array([0, 2.5, 2.5, 5, 5]) if length = 5
    """
    # x = np.array(data)
    x = np.array(data)
    old_length = len(x)
    x = np.repeat(x, length // x.shape[0]).reshape(old_length, -1)
    x = x / x.shape[1]
    x = x.flatten()
    return np.array(x)


def _interpolate_data_linear(data, length):
    """Interpolates data to a desired length using linear interpolation"""
    x = np.linspace(0, len(data) - 1, num=len(data))
    x_new = np.linspace(0, len(data) - 1, num=length)
    return np.interp(x_new, x, data)


def _interpolate_data_stepwise(data, length):
    """
    Interpolates data to a desired length using stepwise interpolation
    new values are set to the previous value
    """
    x = np.array(data)
    x = np.repeat(x, length // x.shape[1], axis=1)
    assert x.shape[1] == length and x.shape[0] == len(data)
    return x


def _get_scenario(dataset: str, average_cars_per_day, minutes_per_timestep):
    def _load_scenario_data(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            column_names = reader.fieldnames
            if dataset not in column_names:
                raise ValueError(f"Dataset '{dataset}' not found in the CSV file")
            data = [float(row[dataset]) for row in list(reader)]
        return jnp.array(data)

    resources = r.files("chargax")
    csv_files = [
        "car_arrival_percentages_workdays.csv",
        "car_arrival_percentages_weekends.csv",
        "car_connection_times.csv",
        "car_energy_demand.csv",
    ]
    data = [
        _load_scenario_data(resources.joinpath(DATA_FOLDER, file)) for file in csv_files
    ]
    desired_length = 24 * 60 // minutes_per_timestep
    data[0] = (
        _average_data(data[0], desired_length) / 100
    ) * average_cars_per_day  # data is in percentages (0-100) --> make it absolute
    data[1] = (
        _average_data(data[1], desired_length) / 100
    ) * average_cars_per_day  # data is in percentages (0-100) --> make it absolute
    data[2] = data[2] * 60  # convert hours to minutes
    return tuple(data)


def _get_car_data(dataset: Literal["eu", "us", "world"]):
    resources = r.files("chargax")
    csv_file = resources.joinpath(DATA_FOLDER, "car_frequency_and_profiles.csv")
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = np.array([list(map(float, row[1:])) for row in list(reader)[1:]])
    profiles = data[:, 3:]
    if dataset == "eu":
        frequency = data[:, 0]
    elif dataset == "us":
        frequency = data[:, 1]
    elif dataset == "world":
        frequency = data[:, 2]

    # make sure frequency is normalized to 1.0
    frequency /= np.sum(frequency)

    # merge the frequency and profiles
    merged_data = np.concatenate((frequency[:, None], profiles), axis=1)

    # remove the rows where the frequency is 0
    merged_data = merged_data[merged_data[:, 0] != 0]

    return merged_data


def _default_get_num_cars_arriving(
    key: PRNGKeyArray, state: EnvState, data: tuple[np.ndarray, np.ndarray]
) -> int:
    """Default function to get the number of cars arriving at the station."""

    arrival_means = jax.lax.select(
        state.is_workday,
        data[0][:, state.timestep],
        data[1][:, state.timestep],
    )
    randint = jax.random.randint(key, (), 0, arrival_means.shape[0])
    random_arrivals = arrival_means[randint]
    return random_arrivals


def default_get_num_cars_arriving_constructor(
    env: Chargax,
    average_cars_per_day: int | Literal["low", "medium", "high"],
    user_profile: Literal["highway", "residential", "workplace", "shopping"],
    seed: int = 0,
) -> Callable[[PRNGKeyArray, EnvState], int]:
    if average_cars_per_day in ["low", "medium", "high"]:
        if average_cars_per_day == "low":
            average_cars_per_day = 50
        elif average_cars_per_day == "medium":
            average_cars_per_day = 100
        elif average_cars_per_day == "high":
            average_cars_per_day = 250

    arrival_data_workdays, arrival_data_weekends, _, _ = _get_scenario(
        user_profile,
        average_cars_per_day=average_cars_per_day,
        minutes_per_timestep=env.minutes_per_timestep,
    )

    def _pre_sample_poisson(key, arrival_means, num_samples=1000):
        """Poission is expensive to sample mid-simulation, so we pre-sample it"""
        data = jax.random.poisson(
            key, lam=arrival_means, shape=(num_samples, len(arrival_means))
        )
        return data

    key1, key2 = jax.random.split(jax.random.PRNGKey(seed), 2)
    weekday_data = _pre_sample_poisson(key1, arrival_data_workdays)
    weekend_data = _pre_sample_poisson(key2, arrival_data_weekends)
    stacked_data = (weekday_data, weekend_data)

    return lambda key, state: _default_get_num_cars_arriving(key, state, stacked_data)


def _default_fill_EVSES_from_scenario(
    key: PRNGKeyArray, state: EnvState, data: EVSE
) -> EVSE:
    """Default function to fill the EVSEs based on the scenario (user and car profiles)"""
    # data is an EVSE with each property in shape of (num_samples, num_chargers)
    # Sample a random sample for each charger and return a EVSE with each property in shape of (num_chargers,)
    num_samples = data.car_battery_capacity_kw.shape[0]
    num_chargers = data.car_battery_capacity_kw.shape[1]
    random_indices = jax.random.randint(key, (num_chargers,), 0, num_samples)
    charger_indices = jnp.arange(num_chargers)

    def _sample(x):
        if hasattr(x, "ndim") and x.ndim == 2:
            return x[random_indices, charger_indices]
        return x

    return jax.tree.map(_sample, data)


def default_fill_EVSES_from_scenario_constructor(
    env: Chargax,
    car_profile: Literal["eu", "us", "world"],
    user_profile: Literal["highway", "residential", "workplace", "shopping"],
    seed: int = 0,
) -> Callable[[PRNGKeyArray], EVSE]:
    """Default function to fill the EVSEs based on the scenario (user and car profiles)"""

    key = jax.random.PRNGKey(seed)
    car_data = jnp.array(_get_car_data(car_profile))
    car_frequencies = car_data[:, 0]
    car_profiles = car_data[:, 1:]

    N_PRESAMPLE = 1000
    num_chargers = env.station.num_chargers
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    car_indices = jax.random.categorical(
        k1, car_frequencies, shape=(N_PRESAMPLE, num_chargers)
    )
    car_profiles_sampled = car_profiles[car_indices]

    _, _, connection_times, energy_demands = _get_scenario(
        user_profile, 100, env.minutes_per_timestep
    )
    connection_times_sampled = connection_times[
        jax.random.randint(k2, (N_PRESAMPLE, num_chargers), 0, len(connection_times))
    ]
    energy_demands_sampled = energy_demands[
        jax.random.randint(k3, (N_PRESAMPLE, num_chargers), 0, len(energy_demands))
    ]
    car_desired_battery_percentage = jax.random.uniform(
        k4, (N_PRESAMPLE, num_chargers), minval=0.8, maxval=0.95
    )
    car_desired_kw = car_profiles_sampled[..., 1] * car_desired_battery_percentage
    car_battery_now_kw = car_desired_kw - energy_demands_sampled
    car_battery_now_kw = jnp.clip(
        car_battery_now_kw,
        0.03 * car_profiles_sampled[..., 1],
        car_profiles_sampled[..., 1],
    )
    if user_profile == "highway":
        charge_sensitive = jax.random.bernoulli(
            k5, 0.9, shape=(N_PRESAMPLE, num_chargers)
        )
    else:
        charge_sensitive = jax.random.bernoulli(
            k5, 0.1, shape=(N_PRESAMPLE, num_chargers)
        )

    presampled_flat_evse = env.station.zero_grid().evses_flat.replace(
        car_ac_absolute_max_charge_rate_kw=car_profiles_sampled[..., 2],
        car_ac_optimal_charge_threshold=car_profiles_sampled[..., 0],
        car_dc_absolute_max_charge_rate_kw=car_profiles_sampled[..., 3],
        car_dc_optimal_charge_threshold=car_profiles_sampled[..., 0],
        car_battery_capacity_kw=car_profiles_sampled[..., 1],
        car_time_till_leave=connection_times_sampled.astype(int),
        car_battery_now_kw=car_battery_now_kw,
        car_desired_battery_percentage=car_desired_battery_percentage,
        charge_sensitive=charge_sensitive,
        car_arrival_battery_kw=car_battery_now_kw,
    )

    return lambda key, _: _default_fill_EVSES_from_scenario(
        key, _, presampled_flat_evse
    )


# def sample_cars(self, key: PRNGKeyArray) -> EVSE:
#         """
#         Returns a ChargersState based on the current scenario (user and car profiles)
#         In the returned ChargersState, all connections are filled
#         with is_car_connected to false. The required number of chargers should then
#         be connected and then merged with the current ChargersState.
#         """
#         chargers_state = self.station.zero_grid().evses_flat
#         if self.car_profiles in ["eu", "us", "world"]:
#             car_data = jnp.array(get_car_data(self.car_profiles))
#             probs = car_data[:, 0]
#             cars = jax.random.categorical(
#                 key, probs, shape=(self.station.num_chargers,)
#             )
#             tau_car_data = car_data[:, 1]
#             capacity_car_data = car_data[:, 2]
#             ac_max_rate_car_data = car_data[:, 3]
#             dc_max_rate_car_data = car_data[:, 4]
#             chargers_state = chargers_state.replace(
#                 car_ac_absolute_max_charge_rate_kw=ac_max_rate_car_data[cars],
#                 car_ac_optimal_charge_threshold=tau_car_data[cars],
#                 car_dc_absolute_max_charge_rate_kw=dc_max_rate_car_data[cars],
#                 car_dc_optimal_charge_threshold=tau_car_data[cars],
#                 car_battery_capacity_kw=capacity_car_data[cars],
#             )

#         if self.user_profiles in ["highway", "residential", "workplace", "shopping"]:
#             _, _, connection_times, energy_demands = get_scenario(self.user_profiles)
#             keys = jax.random.split(key, 3)
#             connection_times_rnd = jax.random.randint(
#                 keys[0], (self.station.num_chargers,), 0, 101
#             )
#             energy_demands_rnd = jax.random.randint(
#                 keys[1], (self.station.num_chargers,), 0, 101
#             )
#             car_time_till_leave = connection_times[connection_times_rnd].astype(int)

#             energy_demands = energy_demands[energy_demands_rnd]
#             car_desired_battery_percentage = jax.random.uniform(
#                 keys[2], (self.station.num_chargers,), minval=0.8, maxval=0.95
#             )
#             car_desired_kw = (
#                 chargers_state.car_battery_capacity_kw * car_desired_battery_percentage
#             )
#             car_battery_now_kw = car_desired_kw - energy_demands
#             car_battery_now_kw = jnp.clip(
#                 car_battery_now_kw,
#                 0.03 * chargers_state.car_battery_capacity_kw,
#                 chargers_state.car_battery_capacity_kw,
#             )

#             if self.user_profiles == "highway":
#                 charge_sensitive = jax.random.bernoulli(
#                     keys[2], 0.9, shape=(self.station.num_chargers,)
#                 )
#             else:
#                 charge_sensitive = jax.random.bernoulli(
#                     keys[2], 0.1, shape=(self.station.num_chargers,)
#                 )

#             chargers_state = chargers_state.replace(
#                 car_time_till_leave=car_time_till_leave,
#                 car_battery_now_kw=car_battery_now_kw,
#                 car_desired_battery_percentage=car_desired_battery_percentage,
#                 charge_sensitive=charge_sensitive,
#             )

#         return chargers_state.replace(
#             car_arrival_battery_kw=chargers_state.car_battery_now_kw,  # copy the initial battery percentage
#         )
