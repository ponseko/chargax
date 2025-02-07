from importlib import resources as r
import numpy as np
import csv

DATA_FOLDER = "data"

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
    return data

def get_scenario(dataset: str, minutes_per_timestep: int = 5):
    resources = r.files("chargax")
    if dataset == "office":
        csv_file = resources.joinpath(DATA_FOLDER, "office_distribution_means.csv")
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = [float(row[1]) for row in list(reader)[1:]]
    desired_length = 24 * 60 // minutes_per_timestep
    return _interpolate_data_linear(data, desired_length)

def get_electricity_prices(dataset: str = "NL", minutes_per_timestep: int = 5):
    resources = r.files("chargax")
    csv_file = resources.joinpath(DATA_FOLDER, f"electricity_prices_kwh_{dataset}.csv")
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = [list(map(float, row[1:])) for row in list(reader)[1:]]
    desired_length = 24 * 60 // minutes_per_timestep
    return _interpolate_data_stepwise(data, desired_length)
