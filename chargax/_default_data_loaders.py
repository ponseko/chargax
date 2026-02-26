from __future__ import annotations

import csv
from importlib import resources as r
from typing import TYPE_CHECKING, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

if TYPE_CHECKING:
    from .chargax import EVSE, Chargax, EnvState

DATA_FOLDER = "data"


def _average_data(data, length):
    """
    Average over data to a desired length
    i.e. array([0, 5, 10]) -> array([0, 2.5, 2.5, 5, 5]) if length = 5
    """
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


def _load_scenario_csvs(dataset: str, average_cars_per_day, minutes_per_timestep):
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


def _load_car_profiles(dataset: Literal["eu", "us", "world"]):
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
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # make sure frequency is normalized to 1.0
    frequency /= np.sum(frequency)

    # merge the frequency and profiles
    merged_data = np.concatenate((frequency[:, None], profiles), axis=1)

    # remove the rows where the frequency is 0
    merged_data = merged_data[merged_data[:, 0] != 0]

    return merged_data


def build_default_scenario(
    env: Chargax,
    car_profile: Literal["eu", "us", "world"] = "eu",
    user_profile: Literal[
        "highway", "residential", "workplace", "shopping"
    ] = "highway",
    average_cars_per_day: int | Literal["low", "medium", "high"] = "medium",
    seed: int = 0,
) -> tuple[
    Callable[[PRNGKeyArray, EnvState], int], Callable[[PRNGKeyArray, EnvState], EVSE]
]:
    """Construct both the car arrival and EVSE filling callables from a single scenario.

    Returns:
        A tuple of (get_num_cars_arriving, get_new_cars_arriving) callables.
    """
    if average_cars_per_day in ["low", "medium", "high"]:
        if average_cars_per_day == "low":
            average_cars_per_day = 50
        elif average_cars_per_day == "medium":
            average_cars_per_day = 100
        elif average_cars_per_day == "high":
            average_cars_per_day = 250

    arrival_data_workdays, arrival_data_weekends, connection_times, energy_demands = (
        _load_scenario_csvs(
            user_profile,
            average_cars_per_day=average_cars_per_day,
            minutes_per_timestep=env.minutes_per_timestep,
        )
    )

    # --- Car arrival callable ---
    N_PRESAMPLE = 1000
    key = jax.random.PRNGKey(seed)
    key, k_wd, k_we, k1, k2, k3, k4, k5 = jax.random.split(key, 8)

    def _pre_sample_poisson(key, arrival_means, num_samples=N_PRESAMPLE):
        """Poisson is expensive to sample mid-simulation, so we pre-sample it"""
        return jax.random.poisson(
            key, lam=arrival_means, shape=(num_samples, len(arrival_means))
        )

    weekday_data = _pre_sample_poisson(k_wd, arrival_data_workdays)
    weekend_data = _pre_sample_poisson(k_we, arrival_data_weekends)
    stacked_data = (weekday_data, weekend_data)

    def _sample_arrivals(key, state):
        arrival_means = jax.lax.select(
            state.is_workday,
            stacked_data[0][:, state.timestep],
            stacked_data[1][:, state.timestep],
        )
        randint = jax.random.randint(key, (), 0, arrival_means.shape[0])
        return arrival_means[randint]

    # --- EVSE filling callable ---
    car_data = jnp.array(_load_car_profiles(car_profile))
    car_frequencies = car_data[:, 0]
    car_profiles = car_data[:, 1:]

    num_chargers = env.station.num_chargers
    car_indices = jax.random.categorical(
        k1, car_frequencies, shape=(N_PRESAMPLE, num_chargers)
    )
    car_profiles_sampled = car_profiles[car_indices]

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

    presampled_flat_evse = env.station.evses_flat.replace(
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

    def _sample_incoming_cars(key, state):
        num_samples = presampled_flat_evse.car_battery_capacity_kw.shape[0]
        n_chargers = presampled_flat_evse.car_battery_capacity_kw.shape[1]
        random_indices = jax.random.randint(key, (n_chargers,), 0, num_samples)
        charger_indices = jnp.arange(n_chargers)

        def _sample(x):
            if hasattr(x, "ndim") and x.ndim == 2:
                return x[random_indices, charger_indices]
            return x

        return jax.tree.map(_sample, presampled_flat_evse)

    return _sample_arrivals, _sample_incoming_cars


def build_default_grid_price_fn(
    env: Chargax, dataset: str = "2023_NL", offset: float = 0
) -> Callable[[EnvState], float]:
    """Default function to get the electricity prices based on the dataset"""

    resources = r.files("chargax")
    csv_file = resources.joinpath(DATA_FOLDER, f"electricity_prices_kwh_{dataset}.csv")
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = [list(map(float, row[1:])) for row in list(reader)[1:]]
    desired_length = 24 * 60 // env.minutes_per_timestep
    data = jnp.array(_interpolate_data_stepwise(data, desired_length))

    return lambda state: data[state.day_of_year][state.timestep] + offset


def build_leave_cars_fn() -> Callable[[PRNGKeyArray, EVSE], Array]:
    """Default function to determine which cars leave the station at each timestep
    This default function is based on the car providing either its desired departure time or
    its desired battery percentage, and whether the car is charge sensitive or time sensitive.
    """

    def _leave_cars(key: PRNGKeyArray, ports: EVSE) -> Array:
        ports.car_time_till_leave
        ports.car_time_waited

        is_leaving_time_sensitive = ports.car_time_till_leave <= 0
        is_leaving_charge_sensitive = ports.car_battery_desired_remaining <= 0
        is_leaving = (is_leaving_charge_sensitive * ports.charge_sensitive) + (
            is_leaving_time_sensitive * ~ports.charge_sensitive
        )
        is_leaving = is_leaving * ports.charger_is_car_connected
        return is_leaving.astype(bool)

    return _leave_cars
