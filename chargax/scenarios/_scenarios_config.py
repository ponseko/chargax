import jax
import jax.numpy as jnp


from chargax import Chargax, ChargingStation
from ._scenarios_station_layout import init_default_homecharger, init_default_businessdistrict_station, init_default_shopping_station

# def make_station(num_chargers):
#     """Build a charging station with the given number of chargers.

#     Replace the inside of this function with your real ChargingStation
#     constructor call, e.g.:
#         return ChargingStation.make(num_chargers=num_chargers, ...)
#     """
#     raise NotImplementedError(
#         f"Wire this up to your real ChargingStation constructor "
#         f"(num_chargers={num_chargers})"
#     )


# One entry per scenario. Each entry is just the ingredients needed to build
# a Chargax environment: how many chargers, what kind of cars/users, and any
# extra settings specific to that scenario.

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


#Buy and selling prices

def my_buy_price(state):
    """Time-of-use pricing: expensive during the day, cheap at night."""
    hour = (state.timestep * 5) / 60.0

    return jnp.where((hour >= 8) & (hour < 20), 0.90, 0.10)

def my_sell_price(state):
    return my_buy_price(state) - 0.05

def no_profit(state):
    return 0.75

SCENARIOS = {
    "basic": { #Wan et al. 2019 
        "station": init_default_homecharger(),
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "residential",
            "average_cars_per_day": "home",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0.05,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging": False,
            "renormalize_currents": True,
            "price_hour_lookahead": 0, #In original paper, they use 24 hours historic data
            "price_hour_lookback": 24,
            "capacity_exceeded_alpha": 0,
            "charged_satisfaction_alpha": 0,
            "battery_degradation_alpha": 0,
        },
    },
    "home": { #Wan et al. 2019 
        "station": init_default_homecharger(),
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "residential",
            "average_cars_per_day": "home",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        "get_num_cars_arriving": build_gaussian_arrival_fn(
        minutes_per_timestep=5, mean_hour=18.0, std_hour=1.0
            ),
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging": True,
            "renormalize_currents": True,
            "price_hour_lookahead": 12, #In original paper, they use 24 hours historic data
            "capacity_exceeded_alpha": 0,
            "charged_satisfaction_alpha": 0.01,
            "battery_degradation_alpha": 1.0,
        },
    },
    "residential": {
        "station": ChargingStation.init_default_station(),        
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "residential",
            "average_cars_per_day": "medium",
            # "grid_price_dataset": "2023_NL",
            # "grid_sell_margin": 0,
        },
        "get_grid_buy_price": no_profit,
        "get_grid_sell_price" : no_profit,
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging": True,
            "renormalize_currents": True,
            "elec_customer_sell_price": 0.25,
            "price_hour_lookahead": 12, #In original paper, they use 24 hours historic data
            "capacity_exceeded_alpha": 1.0,
            "charged_satisfaction_alpha": 1.0,
            "battery_degradation_alpha": 1.0,
        },
    },
    "workplace": { #Combination of CAO et al. 2021 and Jiang et al. 2022
        "station": init_default_businessdistrict_station(), 
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "workplace",
            "average_cars_per_day": "high",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging" : False,
            "renormalize_currents": False,
            "price_hour_lookahead": 12,
            "capacity_exceeded_alpha": 1.0,
            "charged_satisfaction_alpha": 1.0,
            "battery_degradation_alpha": 0,
        },
    },
    "shopping": { #Based on Ponse et al. 2022
       "station": ChargingStation.init_default_station(), 
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "shopping",
            "average_cars_per_day": "medium",
            "grid_price_dataset": "2023_NL",       
            "grid_sell_margin": 0.05,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging": True,
            "renormalize_currents": True,
            "price_hour_lookahead": 12,
            "charged_satisfaction_alpha" : 1,
        },
    },
    "highway": { 
        "station": ChargingStation.init_default_station(), 
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "highway",
            "average_cars_per_day": "high",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging": True,
            "renormalize_currents": True,
            "price_hour_lookahead": 12,
            "capacity_exceeded_alpha": 1.0,
            "charged_satisfaction_alpha": 1.0,
            "battery_degradation_alpha": 1.0,
        },
    },
}

def build_scenario(name):
    """Build and return a ready-to-use Chargax environment for one scenario."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS)}")

    scenario = SCENARIOS[name]

    return Chargax(
        station=scenario["station"],
        default_data_kwargs=scenario["default_data_kwargs"],
        get_grid_buy_price=scenario.get("get_grid_buy_price"),
        get_grid_sell_price=scenario.get("get_grid_sell_price"),
        get_num_cars_arriving=scenario.get("get_num_cars_arriving"),
        **scenario["extra_kwargs"],
    )


def list_scenarios():
    """Names of all available benchmark scenarios."""
    return list(SCENARIOS)
