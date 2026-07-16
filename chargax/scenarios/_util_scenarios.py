import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def build_gaussian_arrival_fn(minutes_per_timestep, mean_hour=18.0, std_hour=1.0):
    """Builds a get_num_cars_arriving function where exactly one car arrives per day,
    at a time drawn from N(mean_hour, std_hour^2).
    """
    timesteps_per_day = (24 * 60) // minutes_per_timestep

    def get_num_cars_arriving(key, state):
        # Fold in day_of_year so the sampled arrival time is fixed for the whole day,
        # regardless of which per-step key is passed in.
        day_key = jax.random.fold_in(key, state.day_of_year)
        arrival_hour = mean_hour + std_hour * jax.random.normal(day_key)
        arrival_timestep = jnp.clip(
            jnp.round(arrival_hour * 60 / minutes_per_timestep),
            0,
            timesteps_per_day - 1,
        ).astype(int)
        return jnp.where(state.timestep == arrival_timestep, 1, 0)

    return get_num_cars_arriving

def build_gaussian_daily_profile_fn(
    minutes_per_timestep: int,
    peak_hour: float = 13.0,
    std_hour: float = 2.5,
    peak_kw: float = 4.0,
    min_kw: float = 0.0,
    supplies_power: bool = False,
):
    """Builds a Callable[[EnvState], float] modeling a daily power profile as a
    Gaussian curve over the hour of day: centered at `peak_hour`, scaled so its
    maximum magnitude equals `peak_kw`, and floored at `min_kw`.

    Use for both PV production and base load.
    """
    peak_density = 1.0 / (std_hour * jnp.sqrt(2 * jnp.pi))
    sign = -1.0 if supplies_power else 1.0

    def profile(state):
        hour = (state.timestep * minutes_per_timestep) / 60.0
        density = jnp.exp(-0.5 * ((hour - peak_hour) / std_hour) ** 2) / (
            std_hour * jnp.sqrt(2 * jnp.pi)
        )
        magnitude = jnp.maximum(peak_kw * (density / peak_density), min_kw)
        return sign * magnitude

    return profile

#Buy and selling prices

def my_buy_price(state):
    """Time-of-use pricing: expensive during the day, cheap at night."""
    hour = (state.timestep * 5) / 60.0

    return jnp.where((hour >= 8) & (hour < 20), 0.90, 0.10)

def my_sell_price(state):
    return my_buy_price(state) - 0.05

def no_profit(state):
    return 0.75

def plot_gaussian_profiles(
    pv_profile,
    base_profile,
    minutes_per_timestep,
    title="Gaussian daily profiles: PV production vs. base load",
    figsize=(10, 5.5),
    save_path=None,
):
    """Plot a PV production profile and a base load profile over 24 hours.
    """
    timesteps_per_day = (24 * 60) // minutes_per_timestep
    timesteps = np.arange(timesteps_per_day)
    hours = timesteps * minutes_per_timestep / 60.0

    pv_values = np.array([pv_profile(t) for t in timesteps])
    base_values = np.array([base_profile(t) for t in timesteps])
    net_values = pv_values + base_values

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(hours, pv_values, label="PV production (supplying)", color="#e8a33d", linewidth=2.2)
    ax.plot(hours, base_values, label="Base load (draining)", color="#3d6fe8", linewidth=2.2)
    ax.plot(hours, net_values, label="Net throughput (base + pv)", color="#444444", linewidth=1.6, linestyle="--")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(hours, pv_values, 0, color="#e8a33d", alpha=0.15)
    ax.fill_between(hours, base_values, 0, color="#3d6fe8", alpha=0.15)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)\n(+ draining / - supplying)")
    ax.set_title(title)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig, ax


# minutes_per_timestep = 5

# pv_profile = build_gaussian_daily_profile_fn(
#     minutes_per_timestep=minutes_per_timestep, peak_hour=13.0, std_hour=2.5,
#     peak_kw=200.0, supplies_power=True,
# )
# base_profile = build_gaussian_daily_profile_fn(
#     minutes_per_timestep=minutes_per_timestep, peak_hour=19.0, std_hour=4.0,
#     peak_kw=30, min_kw=0.15,
# )

# fig, ax = plot_gaussian_profiles(pv_profile, base_profile, minutes_per_timestep)
# plt.show()