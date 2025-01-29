from chargax import EnvState, Chargers, ChargerGroup
import equinox as eqx
import numpy as np

def create_uniform_topology(num_chargers=10, chargers_per_group=2):
    """
        Creates a charger setup with n unfirom chargers distributed 
        over a depth of m levels.
    """
    assert num_chargers % chargers_per_group == 0, "Chargers must be divisible by chargers_per_group"
    assert chargers_per_group >= 1, "Chargers per group must be greater than 0"
    assert num_chargers > chargers_per_group, "Chargers must be greater than chargers_per_group"

    
    default_charger_max_rate = 200.0

    charger_indices = np.arange(num_chargers)
    charger_indices = charger_indices.reshape(-1, chargers_per_group)

    charge_groups = [
        ChargerGroup(
            connections=[ci],
            group_capacity_max=default_charger_max_rate
        ) for ci in charger_indices
    ]
    
    combined_total_capacity = sum([group.group_capacity_max for group in charge_groups])
    grid_connection_node = ChargerGroup(
        connections=charge_groups,
        group_capacity_max=combined_total_capacity
    )

    return grid_connection_node

# def create_uniform_topology_old(chargers=10, chargers_per_group=2):
#     """
#         Creates a charger setup with n unfirom chargers distributed 
#         over a depth of m levels.
#     """
#     assert chargers % chargers_per_group == 0, "Chargers must be divisible by chargers_per_group"
#     assert chargers_per_group >= 1, "Chargers per group must be greater than 0"
#     assert chargers > chargers_per_group, "Chargers must be greater than chargers_per_group"

    
#     default_charger_max_rate = 50.0
#     default_charge_group_max_capacity = default_charger_max_rate * chargers_per_group

#     charge_groups = [
#         ChargerGroup(
#             connections=[ChargerState() for _ in range(chargers_per_group)],
#             group_capacity_max=default_charge_group_max_capacity
#         ) for _ in range(chargers // chargers_per_group)
#     ]
    
#     combined_total_capacity = sum([group.group_capacity_max for group in charge_groups])
#     grid_connection_node = ChargerGroup(
#         connections=charge_groups,
#         group_capacity_max=combined_total_capacity
#     )

#     return grid_connection_node