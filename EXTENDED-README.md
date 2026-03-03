# Chargax - Extended Documentation

## 📦 Installation & Quick start

For those using [uv](https://docs.astral.sh/uv/getting-started/installation/), it is possible to run a standard PPO implementation with default settings by directly running `uv run main.py`.

```bash
git clone git@github.com:ponseko/chargax.git
cd chargax
uv run main.py
```

Alternatively, install the project as an editable package in your favourite virtual environment software. E.g. using conda:

```bash
git clone git@github.com:ponseko/chargax.git
cd chargax
conda create -n chargax python=3.11
conda activate chargax
pip install -e .

python main.py
```

For CUDA support, additionally run `pip install jax[cuda]`.


## 🏗️ Customizing the Station Layout

The main function uses a default charging station initialized as follows:

```python
	import jax
	from chargax import Chargax, ChargingStation
	
	station = ChargingStation.init_default_station()
	env = Chargax(station=station)
	
	key = jax.random.PRNGKey(0)
	obs, state = env.reset_env(key)
```

The station layout can easily be changed. The station is a tree of nodes. You compose it from the following building blocks:

| Node | Purpose |
|---|---|
| `StationSplitter` | Switchboards, cables, transformers — anything that splits or limits power |
| `EVSE` | A group of physical charger connectors |
| `StationBattery` | On-site battery storage |
| `ChargingStation` | Convenience class for the top node of the tree. Essentially just a `StationSplitter` with a constructor and a custom init function. |

These elements require the following properties on initialization:

| Property | Explanation |
|---|---|
| `max_kw_throughput` | The maximum power in kW that can flow through this node. If Chargax is simulated with `renormalize_currents=True` (default), all nodes are normalized to this level at each step. |
| `efficiency` | Multiplier for the actual power that is effectively flowing through this node. |
| `connections` (StationSplitter) | A list of other nodes that are children of this node |
| `num_chargers` (EVSE) | Number of cars that can be connected to this EVSE simultaneously. If multiple cars are connected, they have to share the `max_kw_throughput` |
| `voltage` (EVSE) | Voltage of this EVSE (assumed fixed). Multiplied with `max_current` to calculate `max_kw_throughput` |
| `max_current` (EVSE) | Maximum current that can be set by an agent on each charger of this EVSE. Multiplied with `voltage` to calculate `max_kw_throughput` |
| `capacity_kw` (StationBattery) | Maximum capacity that can be stored in this on-site battery. |

As an example, note the following station layout with 4 EVSEs (2 fast, 2 slow, each with 2 connectors), and a battery on-site. The tree can be nested arbitrarily deep — any `StationSplitter` can contain other splitters, EVSEs, or batteries.
- **`ChargingStation`** (Grid limit: 500 kW)
  - **`StationSplitter`** (Fast Hub: 300 kW)
    - **`EVSE`** (2 DC Chargers: ~150 kW)
    - **`EVSE`** (2 DC Chargers: ~150 kW)
  - **`StationSplitter`** (Slow Hub: 20 kW)
    - **`EVSE`** (2 AC Chargers: ~7.4 kW)
    - **`EVSE`** (2 AC Chargers: ~7.4 kW)
  - **`StationBattery`** (500 kWh capacity, 100 kW limit)

This station can easily be constructed:

```python
from chargax import ChargingStation, StationSplitter, EVSE, StationBattery

station = ChargingStation(
    max_kw_throughput=500.0,  # Grid connection limit
    efficiency=1.0,
    connections=[
        StationSplitter(
            max_kw_throughput=300.0,
            efficiency=0.995,
            # 2 Fast chargers:
            connections=[
                EVSE(num_chargers=2, voltage=600, max_current=250, efficiency=0.995),
                EVSE(num_chargers=2, voltage=600, max_current=250, efficiency=0.995),
            ]
        ),
        StationSplitter(
            max_kw_throughput=300.0,
            efficiency=0.995,
            # 2 Slow chargers:
            connections=[
                EVSE(num_chargers=2, voltage=230, max_current=32, efficiency=0.995),
                EVSE(num_chargers=2, voltage=230, max_current=32, efficiency=0.995),
            ]
        ),
        # 1 on-site battery
        StationBattery(
            capacity_kw=500.0,
            max_kw_throughput=100.0,
            efficiency=0.995,
        ),
    ],
)

env = Chargax(station=station)
```

Likewise, we can construct a large 32 EVSE charging station:


```python
from chargax import ChargingStation, StationSplitter, EVSE, StationBattery

station = ChargingStation(
    max_kw_throughput=3000.0,  # Grid connection limit
    efficiency=1.0,
    connections=[
        StationSplitter(
            max_kw_throughput=5000.0,
            efficiency=0.995,
            # 32 Fast chargers:
            connections=[
                EVSE(num_chargers=2, voltage=600, max_current=250, efficiency=0.995)
                for _ in range(32)
            ]
        ),
        # 1 on-site battery
        StationBattery(
            capacity_kw=5000.0,
            max_kw_throughput=1000.0,
            efficiency=0.995,
        ),
    ],
)

env = Chargax(station=station)
```

## ⚙️ Changing default data loaders

The default car arrivals, car profiles, and grid prices are configured through default_data_kwargs:

```python
env = Chargax(
    station=station,
    default_data_kwargs={
        "car_profile": "us",               # "eu" (default), "us", "world"
        "user_profile": "residential",      # "highway" (default), "residential", "workplace", "shopping"
        "average_cars_per_day": "low",      # "low", "medium", "high" (default) or int
        "grid_price_dataset": "2023_NL",    # Dataset identifier for price data
        "grid_sell_margin": -0.05,          # Sell price offset from buy price
    },
)
```

## 🔌 Injecting Custom Callables

For full control, replace any of the data-generating functions directly. Each callable receives the current environment state, and optionally a PRNG key:

### Custom grid pricing

```python
def my_buy_price(state):
    """Time-of-use pricing: expensive during the day, cheap at night."""
    hour = (state.timestep * env.minutes_per_timestep) / 60.0
    return jnp.where((hour >= 8) & (hour < 20), 0.30, 0.10)

def my_sell_price(state):
    return my_buy_price(state) - 0.05

env = Chargax(
    station=station,
    get_grid_buy_price=my_buy_price,
    get_grid_sell_price=my_sell_price,
)
```

### Custom car arrivals

```python
def my_num_arriving(key, state):
    """Constant arrival rate of 2 cars per timestep."""
    return 2

env = Chargax(
    station=station,
    get_num_cars_arriving=my_num_arriving,
)
```

You can similarly override `get_new_cars_arriving` (generates EVSE entries for new cars) and `get_cars_departing` (determines which cars leave).

## 🎛️ Environment Parameters

The following parameters can be passed to `Chargax(...)` to configure simulation behaviour:

| Parameter | Default | Description |
|---|---|---|
| `minutes_per_timestep` | `5` | Duration of each simulation timestep in minutes. One episode always simulates a single day (24 h), so this also determines the number of steps per episode ($24 \times 60 / \text{minutes\_per\_timestep}$). |
| `num_discretization_levels` | `10` | Number of discrete action levels per charger. E.g. 10 yields actions at 10 %, 20 %, …, 100 % of the maximum rate. |
| `allow_discharging` | `True` | Whether vehicle-to-grid (V2G) discharging is permitted. When enabled, the action space doubles (negative current levels are added). |
| `renormalize_currents` | `True` | Whether to redistribute currents across chargers after each step so that shared capacity constraints (e.g. grid connection limit) are respected. |
| `price_hour_lookahead` | `6` | Number of future hours of electricity prices included in the observation. |
| `elec_customer_sell_price` | `0.75` | Price in €/kWh charged to customers for electricity delivered. |

```python
env = Chargax(
    station=station,
    minutes_per_timestep=15,
    num_discretization_levels=20,
    allow_discharging=False,
    renormalize_currents=True,
    price_hour_lookahead=12,
    elec_customer_sell_price=0.50,
)
```

## 🏆 Reward Function & Alpha Weights

The reward at each timestep is computed as:

$$r_t = \Delta\text{profit} - \left( \alpha_\text{cap} \cdot \Delta\text{exceeded} + \alpha_\text{sat} \cdot \Delta\text{uncharged\_kw} + \alpha_\text{time} \cdot (\Delta\text{overtime} - \beta \cdot \Delta\text{undertime}) + \alpha_\text{rej} \cdot \Delta\text{rejected} + \alpha_\text{bat} \cdot \Delta\text{discharged\_kw} \right)$$

Where each $\Delta$ is the per-step change. Profit is the revenue from selling electricity to customers minus the grid electricity cost. The penalty terms allow you to shape the reward towards different objectives:

| Parameter | Default | Penalises |
|---|---|---|
| `capacity_exceeded_alpha` | `0.0` | Exceeding the station's grid capacity limit (kW) |
| `charged_satisfaction_alpha` | `0.0` | Unmet customer charging demand (uncharged kWh at departure) |
| `time_satisfaction_alpha` | `0.0` | Overtime (car stays too long) and undertime (car leaves early), weighted by `beta` |
| `rejected_customers_alpha` | `0.0` | Customers rejected because no charger was available |
| `battery_degradation_alpha` | `0.0` | Battery degradation, proxied by total discharged kWh |
| `beta` | `0.0` | Discount factor for undertime within the time satisfaction penalty |

All alpha weights default to `0.0`, meaning the reward is purely profit-based unless you opt into additional penalty terms:

```python
env = Chargax(
    station=station,
    capacity_exceeded_alpha=1.0,
    charged_satisfaction_alpha=0.5,
    time_satisfaction_alpha=0.1,
    rejected_customers_alpha=2.0,
    battery_degradation_alpha=0.01,
    beta=0.5,
)
```

## 👁️ Observation Space

The observation returned by `env.reset_env(key)` and each `env.step_env(...)` is a dictionary with the following keys:

| Key | Shape / Type | Description |
|---|---|---|
| `evses` | `EVSE` pytree | Full state of every EVSE, including per-charger arrays for battery level, capacity, time till leave, current, connection status, etc. |
| `batteries` | `StationBattery` pytree | Full state of every on-site battery (current level, throughput, capacity). |
| `future_buy_prices` | `(price_hour_lookahead,)` | Grid buy prices (€/kWh) for the current and next hours. |
| `future_sell_prices` | `(price_hour_lookahead,)` | Grid sell prices (€/kWh) for the current and next hours. |
| `future_price_diffs_buy` | `(price_hour_lookahead - 1,)` | Difference between each future buy price and the current buy price. |
| `future_price_diffs_sell` | `(price_hour_lookahead - 1,)` | Difference between each future sell price and the current sell price. |
| `current_timestep` | `int` | Current timestep index within the day. |
| `current_day_of_year` | `int` | Day of the year (0–364). |
| `is_workday` | `bool` | Whether the current day is a weekday (Mon–Fri). |

The EVSE pytree exposes per-charger arrays with fields such as `car_battery_now_kw`, `car_battery_capacity_kw`, `car_time_till_leave`, `car_desired_battery_percentage`, `charger_current_now`, `charger_is_car_connected`, and others. See the `EVSE` class for the full list.

## 🎮 Action Space

Actions are passed as a dictionary with two keys: `"evses"` and `"batteries"`.

**EVSE actions** — For each EVSE with $n$ chargers, the action is a `MultiDiscrete(n)` array. Each element is an integer in $[0, L]$ where:

- $L = \texttt{num\_discretization\_levels}$ if discharging is disabled
- $L = 2 \times \texttt{num\_discretization\_levels}$ if discharging is enabled

The idle action is the midpoint when discharging is allowed (i.e. action $= L/2$), or `0` when discharging is disabled. Actions below the midpoint correspond to discharging; actions above correspond to charging.

**Battery actions** — For each on-site battery, the action is a single `Discrete` integer in $[0, 2 \times \texttt{num\_discretization\_levels}]$. The midpoint is idle; lower values discharge the battery and higher values charge it.

```python
# Example: inspect the action space
env = Chargax(station=station)
print(env.action_space)
```

## 📊 Logging

Logging happens in the agent. By default, [Jaxnasium](https://github.com/ponseko/jaxnasium) agents are used with some built-in logging functionality. The PPO parameters below will, every 5% of training timesteps, print the episode rewards of the last performed rollouts to stdout.

```python

# Create the environment
env = Chargax(station=charging_station)
env = jym.LogWrapper(env) # Add LogWrapper!

# RL Training with PPO
agent = PPO(log_function="simple", log_interval=0.05)
agent = agent.train(rng, env)

```

You can write any custom log_function and insert it into the agent:

```python

def my_custom_logging_function(
    data, # The info dict of every step of the last rollout is passed into this function
    iteration, # The training iteration is passed here
):
    processed_data = process_data(data)
    wandb.log(processed_data)
    np.save(processed_data, ...)

log_interval = 1 # Log EVERY training iteration (after every rollout)
agent = PPO(log_function=my_custom_logging_function, log_interval=log_interval)
agent = agent.train(rng, env)

```

If you want to include more data in the info dict that you can log, simply subclass Chargax and override the `get_info` function. Note however that including a lot of info in the info dict, increases the memory requirements during training -- especially when training in many environments in parallel and with long rollouts.

```python
class Charging_w_AlteredInfo(Chargax):
    def get_info(
        self, state: EnvState, actions, old_state: EnvState = None
    ) -> Dict[str, Array]:
        return {
            # ... state variables, actions, or state variables from the previous state
        }
```