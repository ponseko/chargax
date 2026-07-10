from abc import abstractmethod
from dataclasses import fields
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from chargax._station_layout import ChargingStation, EVSE, StationBattery, StationSplitter

def init_default_homecharger() -> "ChargingStation":
        """Initializes a station layout with one car.
        Numbers roughly based on Wan et al. (2019).
        """
        return ChargingStation(
            max_kw_throughput=6,  # Grid connection max throughput, based on the max charging rate being 6kw.
            efficiency=1.0,
            connections=[
                        EVSE(
                            voltage=230,
                            max_current=26, # 6000/230 = 26.08
                            num_chargers=1,
                            efficiency=0.995, #Simple scenario, no losses
                        )                                       
            ]
        )


def init_default_shopping_station() -> "ChargingStation":
    """Initializes a station layout with a mix of fast and slow chargers and a battery on site.
    This site has a constrained grid connection and thus requires the battery to meet demand during peak hours.
    """
    return ChargingStation(
        max_kw_throughput=200.0,  # Grid connection max throughput
        efficiency=1.0,
        connections=[
            StationSplitter(
                max_kw_throughput=650.0,
                efficiency=0.995,
                connections=[
                    # Fast charger:
                    StationSplitter(
                        max_kw_throughput=600.0,
                        efficiency=0.995,
                        connections=[
                            EVSE(
                                voltage=600,
                                max_current=500,
                                num_chargers=2,
                                efficiency=0.995,
                            ),
                            EVSE(
                                voltage=600,
                                max_current=500,
                                num_chargers=2,
                                efficiency=0.995,
                            ),
                        ],
                    ),
                    # Slow charger:
                    StationSplitter(
                        max_kw_throughput=50.0,
                        efficiency=0.995,
                        connections=[
                            EVSE(
                                voltage=230,
                                max_current=50,
                                num_chargers=2,
                                efficiency=0.995,
                            )
                        ],
                    ),
                    # Battery on site:
                    StationBattery(
                        capacity_kw=2500.0,
                        max_kw_throughput=500.0,
                        efficiency=0.995,
                    ),
                ],
            ),
        ],
    )


def init_default_businessdistrict_station() -> "ChargingStation":
    """Initializes a station layout with only slow chargers inspired by the business district scenario in Cao et al. 2021 and Jiang et al. 2022.
    """
    return ChargingStation(
        max_kw_throughput=200.0,  # Grid connection max throughput
        efficiency=1.0,
        connections=[
            # Slow charger:
            StationSplitter(
                max_kw_throughput=50.0,
                efficiency=0.995,
                connections=[
                    EVSE(
                        voltage=230,
                        max_current=50,
                        num_chargers=20,
                        efficiency=0.995,
                    )
                ],
            ),
            # # Battery on site:
            # StationBattery(
            #     capacity_kw=2500.0,
            #     max_kw_throughput=500.0,
            #     efficiency=0.995,
            # ),                         
        ],
    )