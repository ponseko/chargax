from chargax import EnvState, ChargersState, StationSplitter

def pretty_print_charger_group(group: StationSplitter, chargers: ChargersState, indent=0, is_last=True, prefix=""):
    indent_str = ' ' * indent
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}ChargerGroup(group_capacity_max_kw={group.group_capacity_max_kw}, group_rate_current={group.group_rate_current(chargers)})")
    
    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, connection in enumerate(group.connections):
        is_last_connection = (i == len(group.connections) - 1)
        if isinstance(connection, StationSplitter):
            pretty_print_charger_group(connection, chargers, indent + 4, is_last_connection, new_prefix)
        else:
            charger_idx = connection
            charger_connector = "└── " if is_last_connection else "├── "
            print(f"{new_prefix}{charger_connector}ChargersState(charge_rate_current={chargers.charger_rate_current[charger_idx]}, car_connected={chargers.car_connected[charger_idx]})")
            for idx in charger_idx:
                if chargers.car_connected[idx]:
                    car_connector = "    └── " if is_last_connection else "│   └── "
                    print(f"{new_prefix}{car_connector}CarState(time_waiting={chargers.time_waiting[idx]}, time_charging={chargers.time_charging[idx]}, time_till_leave={chargers.time_till_leave[idx]}, battery_level={chargers.battery_level[idx]}, battery_capacity={chargers.battery_capacity[idx]}, battery_temperature={chargers.battery_temperature[idx]})")
                else:
                    none_connector = "    └── " if is_last_connection else "│   └── "
                    print(f"{new_prefix}{none_connector}No Car Connected")


# def pretty_print_charger_group(group, indent=0, is_last=True, prefix=""):
#     indent_str = ' ' * indent
#     connector = "└── " if is_last else "├── "
#     print(f"{prefix}{connector}ChargerGroup(group_capacity_max_kw={group.group_capacity_max_kw}, group_rate_current={group.group_rate_current})")
    
#     new_prefix = prefix + ("    " if is_last else "│   ")
#     for i, connection in enumerate(group.connections):
#         is_last_connection = (i == len(group.connections) - 1)
#         if isinstance(connection, ChargerGroup):
#             pretty_print_charger_group(connection, indent + 4, is_last_connection, new_prefix)
#         elif isinstance(connection, ChargersState):
#             charger_connector = "└── " if is_last_connection else "├── "
#             print(f"{new_prefix}{charger_connector}ChargersState(charge_rate_current={connection.charger_rate_current}, car_connected={connection.car_connected})")
#             if connection.car_connected:
#                 car_connector = "    └── " if is_last_connection else "│   └── "
#                 print(f"{new_prefix}{car_connector}CarState(time_waiting={connection.car.time_waiting}, time_charging={connection.car.time_charging}, time_till_leave={connection.car.time_till_leave}, battery_level={connection.car.battery_level}, battery_capacity={connection.car.battery_capacity}, battery_temperature={connection.car.battery_temperature})")
#             else:
#                 none_connector = "    └── " if is_last_connection else "│   └── "
#                 print(f"{new_prefix}{none_connector}No Car Connected")


# def pretty_print_charger_group(group, indent=0, is_last=True, prefix=""):
#     indent_str = ' ' * indent
#     connector = "└── " if is_last else "├── "
#     print(f"{prefix}{connector}ChargerGroup(group_capacity_max_kw={group.group_capacity_max_kw}, group_rate_current={group.group_rate_current})")
    
#     new_prefix = prefix + ("    " if is_last else "│   ")
#     for i, charger in enumerate(group.chargers):
#         is_last_charger = (i == len(group.chargers) - 1)
#         if isinstance(charger, ChargerGroup):
#             pretty_print_charger_group(charger, indent + 4, is_last_charger, new_prefix)
#         else:
#             charger_connector = "└── " if is_last_charger else "├── "
#             print(f"{new_prefix}{charger_connector}ChargersState(charge_rate_max={charger.charge_rate_max}, charge_rate_current={charger.charge_rate_current}, car={charger.car})")