# Chargax: A JAX Accelerated EV Charging Simulator

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)

> [!NOTE]
> Please refer to the submission branch to reproduce results as presented in the paper.
> The main branch ships the bare environment.

---

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

for CUDA support, additionally run `pip install jax[cuda]`.

---

> [!NOTE]
> The PPO implementation used in the example `train.py` file is different from the one
> used in the paper. As such, required hyperparameters may be different as well.
> Check out the submission branch to reproduce results.


### 🏗️ Customizing the Station Layout

The main function uses a default charging station initialized as follows:

```python
	import jax
	from chargax import Chargax, ChargingStation
	
	station = ChargingStation.init_default_station()
	env = Chargax(station=station)
	
	key = jax.random.PRNGKey(0)
	obs, state = env.reset_env(key)
```

The station layout can easily be changed. The station is a tree of nodes. You compose it from three building blocks:

| Node | Purpose |
|---|---|
| `StationSplitter` | Switchboards, cables, transformers — anything that splits or limits power |
| `EVSE` | A group of physical charger connectors |
| `StationBattery` | On-site battery storage |


#### Defining a custom station

```python
from chargax import ChargingStation, StationSplitter, EVSE, StationBattery

station = ChargingStation(
max_kw_throughput=150.0,  # Grid connection limit
efficiency=1.0,
connections=[
      StationSplitter(
            max_kw_throughput=150.0,
            efficiency=0.995,
            connections=[
            # 4 slow AC chargers (2 per EVSE)
            EVSE(num_chargers=2, voltage=230, max_current=32, efficiency=0.995),
            EVSE(num_chargers=2, voltage=230, max_current=32, efficiency=0.995),
            # 1 on-site battery
            StationBattery(
                  capacity_kw=500.0,
                  max_kw_throughput=100.0,
                  efficiency=0.99,
            ),
            ],
      ),
],
)

env = Chargax(station=station)
```

The tree can be nested arbitrarily deep — any StationSplitter can contain other splitters, EVSEs, or batteries.

## ⚙️ Configuring the Environment

#### Changing default data loaders

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

#### 🔌 Injecting Custom Callables

For full control, replace any of the data-generating functions directly. Each callable receives a PRNG key and the current environment state:

##### Custom grid pricing

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

##### Custom car arrivals

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


## 📑 Citing

```bibtex
@misc{ponse2025chargaxjaxacceleratedev,
      title={Chargax: A JAX Accelerated EV Charging Simulator}, 
      author={Koen Ponse, Jan Felix Kleuker, Aske Plaat, Thomas Moerland},
      year={2025},
      eprint={2507.01522},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.01522}, 
}
```
