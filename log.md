
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
- 

