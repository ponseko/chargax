import jax
import jax.numpy as jnp


from chargax import Chargax, ChargingStation
from chargax.scenarios._scenarios_station_layout import init_default_homecharger, init_default_businessdistrict_station, init_default_shopping_station, init_default_basecharger, init_grid_network_station
from chargax._default_data_loaders import build_default_pv_production_fn
from chargax.scenarios._util_scenarios import *

# One entry per scenario. Each entry is just the ingredients needed to build
# a Chargax environment: how many chargers, what kind of cars/users, and any
# extra settings specific to that scenario.


SCENARIOS = {
    "basic": { 
        "station": init_default_basecharger(),
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "residential",
            "average_cars_per_day": "home",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        #     "get_num_cars_arriving": build_gaussian_arrival_fn(
        # minutes_per_timestep=5, mean_hour=18.0, std_hour=1.0
        #     ),
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 2,
            "allow_discharging": False,
            "renormalize_currents": True,
            "price_hour_lookahead": 0, 
            "price_hour_lookback": 0,
            "capacity_exceeded_alpha": 0,
            "charged_satisfaction_alpha": 0,
        },
    },
    "home": { #Wan et al. 2019 
        "station": init_default_homecharger(
                pv_profile= build_gaussian_daily_profile_fn(
                    minutes_per_timestep=5, peak_hour=13.0, std_hour=2.5, peak_kw=4.0,supplies_power=True),
                base_profile= build_gaussian_daily_profile_fn(
                    minutes_per_timestep=5, peak_hour=19.0, std_hour=4.0, peak_kw=0.6, min_kw=0.15,)
                ),
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "residential",
            "average_cars_per_day": "home",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        # "get_num_cars_arriving": build_gaussian_arrival_fn(
        # minutes_per_timestep=5, mean_hour=18.0, std_hour=1.0
        #     ),
        "extra_kwargs": {
            "minutes_per_timestep": 5,
            "num_discretization_levels": 10,
            "allow_discharging": True,
            "renormalize_currents": True,
            "price_hour_lookahead": 0, #In original paper, they use 24 hours historic data
            "capacity_exceeded_alpha": 24,
            "charged_satisfaction_alpha": 0.01,
            "battery_degradation_alpha": 1.0,
        },
    },
    "residential": {
        "station": init_grid_network_station(pv_profile=-100 #build_gaussian_daily_profile_fn(
                    #minutes_per_timestep=5, peak_hour=13.0, std_hour=2.5, peak_kw=200.0,)
                    ),        
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "residential",
            "average_cars_per_day": 750,
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        # "get_grid_buy_price": no_profit,
        # "get_grid_sell_price" : no_profit,
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
    "shopping_pv": { #Based on Ponse et al. 2022
       "station": init_default_shopping_station(                
                pv_profile= build_gaussian_daily_profile_fn(
                    minutes_per_timestep=5, peak_hour=13.0, std_hour=2.5, peak_kw=200.0,),
                base_profile= build_gaussian_daily_profile_fn(
                    minutes_per_timestep=5, peak_hour=19.0, std_hour=4.0, peak_kw=50, min_kw=20,)
                ), 
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
            "capacity_exceeded_alpha": 1.0,
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
            "capacity_exceeded_alpha": 2.0,
            "charged_satisfaction_alpha": 1.0,
            "battery_degradation_alpha": 1.0,
        },
    },
}

def build_scenario(name, overrides: dict | None = None):
    """Build and return a ready-to-use Chargax environment for one scenario."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS)}")

    scenario = SCENARIOS[name]
    extra_kwargs = {**scenario["extra_kwargs"], **(overrides or {})}

    return Chargax(
        station=scenario["station"],
        default_data_kwargs=scenario["default_data_kwargs"],
        get_grid_buy_price=scenario.get("get_grid_buy_price"),
        get_grid_sell_price=scenario.get("get_grid_sell_price"),
        get_num_cars_arriving=scenario.get("get_num_cars_arriving"),
        **extra_kwargs,
    )


def list_scenarios():
    """Names of all available benchmark scenarios."""
    return list(SCENARIOS)


