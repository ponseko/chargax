import distrax
import jax
import numpy as np

from scenarios.scenario_data import office_distribution_means

def interpolate_arrival_data(mean_data, timestep_interval_minutes, total_simulation_days, std_window_size=10):
    total_timesteps = int(24*60*total_simulation_days // timestep_interval_minutes)

    # Interpolate the means to match the total_timesteps
    x = np.linspace(0, len(mean_data) - 1, num=len(mean_data))
    x_new = np.linspace(0, len(mean_data) - 1, num=total_timesteps)
    interpolated_means = np.interp(x_new, x, mean_data)

    def wrapped_std(arr, idx, std_window_size):
        extended_arr = np.concatenate([arr[-std_window_size:], arr, arr[:std_window_size]])
        window_start = idx
        window_end = idx + 2 * std_window_size + 1
        return np.std(extended_arr[window_start:window_end])
    
    interpolated_stds = [wrapped_std(interpolated_means, i, std_window_size) for i in range(total_timesteps)]

    return jax.tree.map(
        lambda mean, std: distrax.Normal(loc=mean, scale=std),
        interpolated_means.tolist(), interpolated_stds
    )
