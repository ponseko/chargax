# List of changes made and implementation added:
- Historic pricing: price_hours_lookback
- Allow bigger timesteps than provided data: changed _average_data to _resample_data
 
# 1-7-2026

General goal of the day was to create a home charging version:

- Fixed a bug in jaxnasium.spaces.Box: removed the squeeze() function so it works with one charger.
- Created the datamanager file to visualise the data and add gaussian distributions. 
Add rows to the different distributions to mimic the home scenario. Residential distribution is accurate for the arrival time in the workweek. 
Added car profile of Nissan Leaf (most used in home scenarios)
- Build the charging station layout.
- Experimented with different objectives. Trying to get the objective charging costs, battery degradation and range anxiety to work.
- Occupancy is very low currently this is probably because there is only one ev coming per day and the connection times are very low. Should be around 14 hours. Should be around 50% rate.
- Fixed that the granularity of the system can also be made bigger than the bins made. 

### TO DO:
- To do run different test by doing a config sweep.
- Create updated data profiles. (connection times need to be longer around 14 hours. (look into modelling this by using the departure time as input))
- Connection time has a mistake both cumsum and not (ask Koen)
- Add discrete and continuous action space.
- Think of what constitutes as a good learning problem. If each version of the problem can be solved by as simple PPO algorithm then what are we doing here. 

# 2-7-2026

Goal of the day is to makes as 5 scenarios based on papers:

- Single car at a home with a base load, objective is to minimize cost, battery degradation and range anxiety. Discharging allowed, includes base load and ev production. Wan et al. 2019
- Shopping scenario in Ponse et al.

Updated the arrival patterns to also include the home scenario.
Made a file to easily run different scenarios.

### TO DO:
- Implement historic pricing data in the state space. 
- Figure out how to include custom functions in the scenario file or in the default data loaders. Maybe overhaul this completely to make more streamlined. 

### Vragen voor Koen: 
- Wat doet capacity exceed? Indicates if the capacity is exceeded, it looks like it does not get overruled when currents are normalised. Might reappear in which case tell Koen. 
- Hoe werkt de connection times? Is het in uren, timesteps of minuten? It works in hours and is cumulative. Fixed a bug related to the highway scenario. 

# 6-7-2026

Moved everything I was working on to the most updated state of chargax/Got access to the GitHub

### TO DO:
- Check if the new system runs the scenarios
- Allow for experimental and modeled input
- Add other scenarios
- Find data for Base load and PV production
- Determine the reported evaluations metrics

# 10-7-2026
Added the PV files. Try to incorporate the PV production in Chargax. 
Similar build as the pricing data

Go over Chargax extensively. Finished reading the Chargax file.
Added the historic pricing

# 13-7-2026
Finished going over the station layout file and the default data loaders file.
Changed _average_data to _resample_data to allow for timesteps that are bigger than the timesteps in the dataset. 

### TO DO:
- Add that baselines measure values other than profit.
- Add PV production data collector
- Add PV production to station layout
- Add Base load callable function
- Add Base load to station layout 
- Add baseline that has a fixed charging rate. 
- Add Polderwijk

# 14-7-2026

Changed load_profile: Callable[[Any], float] | float = 0.0 to load_profile: Callable[[Any], float] | float = eqx.field(static=True, default=0.0)
Check with Koen if this does not have bad consequences.

Check with Koen on how to implement the data based PV production 

Implemented:
- Polderwijk grid (very large residential parking lot with solar and high traffic)
- Base load added to station layout 
- Gaussian function to create energy loads
- Added extra measures to the baselines but have not checked them.

### TO DO:
- Add baseline that has a fixed charging rate. 
- Integrate PV production data collector
- Add the correct PV production, base load, number of cars and lookahead pricing to each scenario
- Fix the reward function (look into removing profit and making it a fixed rate)
- To do run different test by doing a config sweep.
- Determine the reported evaluations metrics

# 16-7-2026

Added the FIxedRateCharge baseline based on the MAxCharge baseline. Interestingly 0.75 charging rate seems to outperform MaxCharge in Profit and rewards. This could be because of the fluctuation in charging prices. 

There is a specific bug that when you set price lookahead or lookback at 1 this gives an error. 

Added a way to run large sweeps of different combination of the scenarios including possible overrides to test different set ups. 

## Basic Scenario
Some interesting results in the basic scenario
For the following setting:
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

Seed 42
avg_reward scenario  ... override__charged_satisfaction_alpha  avg_profit
0    5.462128    basic  ...                                    0         NaN DQN
1    5.462128    basic  ...                                    0         NaN SAC
2    5.462128    basic  ...                                    0         NaN PPO
3    4.999728    basic  ...                                    0  273.048157 Max
4    3.680146    basic  ...                                    0  200.761902 Mid  
5    3.740823    basic  ...                                    0  202.998474 Random

Here we see that for this setting we get that all the different RL methods converge to the same result. 
How can the reward be higher than the reward for MaxProfit? Is this because the grid prices are flexible?

Done several runs with different seeds. All of them converge to roughly the same value each run and outperform the baselines for this simple problem.
This however is with profit. Ideally the profit is minimized if we are thinking from the perspective of the user. 

## Home Scenario

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
    
On the home scenario, where there are other factors involved, we still see the baselines outperformed and seemingly convergence?  

Seed 45
avg_reward scenario              agent                run_name  elapsed_sec  avg_profit
0    6.207622     home                dqn                home-dqn    22.861480         NaN
1    6.207622     home                sac                home-sac    69.757327         NaN
2    5.902182     home                ppo                home-ppo    52.279659         NaN
3    4.772661     home         max_charge         home-max_charge    17.159780  387.409851
4    3.596998     home  fixed_rate_charge  home-fixed_rate_charge    18.218999  291.072235
5   -4.895920     home             random             home-random    19.437305   83.545128

Seed 42
avg_reward scenario              agent                run_name  elapsed_sec  avg_profit
0    5.300367     home                dqn                home-dqn    20.867837         NaN
1    5.304589     home                sac                home-sac    66.592666         NaN
2   -0.000021     home                ppo                home-ppo    32.050279         NaN
3    4.888210     home         max_charge         home-max_charge    19.946924  269.404175
4    3.680146     home  fixed_rate_charge  home-fixed_rate_charge    19.391447  200.761902
5   -3.705442     home             random             home-random    21.541838   54.866005

For seed 42 there is some weird collapse for the ppo 