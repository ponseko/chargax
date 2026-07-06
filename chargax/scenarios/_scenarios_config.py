from chargax import Chargax, ChargingStation


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
SCENARIOS = {
    "home": { #Wan et al. 2019 
        "station": ChargingStation.init_default_homecharger(),
        "default_data_kwargs": {
            "car_profile": "home",
            "user_profile": "home",
            "average_cars_per_day": "home",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 60,
            "num_discretization_levels": 7,
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
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 60,
            "num_discretization_levels": 7,
            "allow_discharging": True,
            "renormalize_currents": True,
            "price_hour_lookahead": 12, #In original paper, they use 24 hours historic data
            "capacity_exceeded_alpha": 1.0,
            "charged_satisfaction_alpha": 1.0,
            "battery_degradation_alpha": 1.0,
        },
    },
    "workplace": {
        "station": ChargingStation.init_default_station(), 
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "workplace",
            "average_cars_per_day": "medium",
            "grid_price_dataset": "2023_NL",
            "grid_sell_margin": 0,
        },
        "extra_kwargs": {
            "minutes_per_timestep": 15,
            "num_discretization_levels": 7,
            "allow_discharging": True,
            "renormalize_currents": True,
            "price_hour_lookahead": 12,
            "capacity_exceeded_alpha": 1.0,
            "charged_satisfaction_alpha": 1.0,
            "battery_degradation_alpha": 1.0,
        },
    },
    "shopping": { #Based on Ponse et al. 2022
       "station": ChargingStation.init_default_station(), 
        "default_data_kwargs": {
            "car_profile": "eu",
            "user_profile": "shopping",
            "average_cars_per_day": "medium",
            "grid_price_dataset": "2023_NL",
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
            "num_discretization_levels": 7,
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
        **scenario["extra_kwargs"],
    )


def list_scenarios():
    """Names of all available benchmark scenarios."""
    return list(SCENARIOS)
