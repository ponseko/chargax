from importlib import resources as r
import numpy as np
import csv
from typing import Literal
import jax.numpy as jnp

DATA_FOLDER = "data"

def _average_data(data, length):
    """ 
        Average over data to a desired length 
        i.e. array([0, 5, 10]) -> array([0, 2.5, 2.5, 5, 5]) if length = 5
    """
    # x = np.array(data)
    x = jnp.array(data)
    old_length = len(x)
    x = jnp.repeat(x, length // x.shape[0]).reshape(old_length, -1)
    x = x / x.shape[1]
    x = x.flatten()
    return jnp.array(x)

def _interpolate_data_linear(data, length):
    """ Interpolates data to a desired length using linear interpolation """
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

# def get_scenario(dataset: str, minutes_per_timestep: int = 5):
#     resources = r.files("chargax")
#     if dataset == "office":
#         csv_file = resources.joinpath(DATA_FOLDER, "office_distribution_means.csv")
#     with open(csv_file, "r") as f:
#         reader = csv.reader(f)
#         data = [float(row[1]) for row in list(reader)[1:]]
#     desired_length = 24 * 60 // minutes_per_timestep
#     return _average_data(data, desired_length)
#     return _interpolate_data_linear(data, desired_length)

def get_scenario(dataset: str, average_cars_per_day: int = 100, minutes_per_timestep: int = 5):
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
        "car_energy_demand.csv"
    ]
    data = [_load_scenario_data(resources.joinpath(DATA_FOLDER, file)) for file in csv_files]
    desired_length = 24 * 60 // minutes_per_timestep
    data[0] = (_average_data(data[0], desired_length) / 100) * average_cars_per_day # data is in percentages (0-100) --> make it absolute
    data[1] = (_average_data(data[1], desired_length) / 100) * average_cars_per_day # data is in percentages (0-100) --> make it absolute
    data[2] = data[2] * 60 # convert hours to minutes
    return tuple(data)

def get_electricity_prices(dataset: str = "2023_NL", minutes_per_timestep: int = 5):
    resources = r.files("chargax")
    csv_file = resources.joinpath(DATA_FOLDER, f"electricity_prices_kwh_{dataset}.csv")
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = [list(map(float, row[1:])) for row in list(reader)[1:]]
    desired_length = 24 * 60 // minutes_per_timestep
    return _interpolate_data_stepwise(data, desired_length)

def get_car_data(dataset: Literal["eu", "us", "world"]):
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
