import csv
import datetime
from importlib import resources

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest

from chargax import Chargax, ChargingStation
from chargax._default_data_loaders import _average_data
from chargax._util import minute_of_day, weekday


@pytest.mark.parametrize(
    "date", ["2020-01-01", "2023-01-01", "2023-11-12", "2024-02-29", "2024-12-31"]
)
def test_weekday_and_minute_of_day_match_python(date):
    dt = jdt.to_datetime(date) + jdt.Timedelta(seconds=13 * 3600 + 25 * 60)
    expected = datetime.datetime.fromisoformat(date)
    assert int(weekday(dt)) == expected.weekday()
    assert int(minute_of_day(dt)) == 13 * 60 + 25


@pytest.mark.parametrize("length", [48, 144, 288])
def test_average_data_preserves_total(length):
    x = np.random.default_rng(0).random(96)
    resampled = _average_data(x, length)
    assert resampled.shape == (length,)
    assert np.isclose(resampled.sum(), x.sum())


@pytest.mark.parametrize("minutes_per_timestep", [5, 30])
def test_grid_price_follows_dataset_over_multiple_days(minutes_per_timestep):
    env = Chargax(
        station=ChargingStation.init_default_station(),
        minutes_per_timestep=minutes_per_timestep,
        simulation_length_days=3,
    )
    idle = jax.tree.map(jnp.zeros_like, env.sample_action(jax.random.PRNGKey(0)))

    @jax.jit
    def rollout(key):
        _, state = env.reset(key)

        def step(state, _):
            price = env.get_grid_buy_price(state)
            _, state = env.step_env(key, state, idle)
            return state, price

        return state, jax.lax.scan(step, state, length=env.max_episode_steps)[1]

    start_state, prices = rollout(jax.random.PRNGKey(3))
    start = start_state.datetime.to_pydatetime()

    csv_file = resources.files("chargax").joinpath(
        "data", "electricity_prices_kwh_2023_NL.csv"
    )
    with csv_file.open() as f:
        rows = list(csv.reader(f))[1:]
    for i in range(0, env.max_episode_steps, 60 // minutes_per_timestep):
        t = start + datetime.timedelta(minutes=i * minutes_per_timestep)
        row = rows[t.timetuple().tm_yday - 1]
        if t.year == 2023:  # the dataset row is this exact date (wraps after Dec 31st)
            assert row[0] == t.strftime("%Y-%m-%d")
        assert prices[i] == pytest.approx(float(row[1 + t.hour]), abs=1e-4), t


def test_arrivals_depend_on_time_of_day_not_episode_step():
    env = Chargax(station=ChargingStation.init_default_station())
    _, state = env.reset(jax.random.PRNGKey(0))
    noon_steps = 12 * 60 // env.minutes_per_timestep
    state = state._replace(
        timestep=state.timestep + noon_steps,
        datetime=state.datetime + jdt.Timedelta(seconds=12 * 3600),
    )
    week_steps = 7 * env.max_episode_steps
    week_later = state._replace(
        timestep=state.timestep + week_steps,
        datetime=state.datetime + jdt.Timedelta(days=7),
    )
    for key in jax.random.split(jax.random.PRNGKey(1), 20):
        assert env.get_num_cars_arriving(key, state) == env.get_num_cars_arriving(
            key, week_later
        )


def _csv_price(date: datetime.datetime) -> float:
    csv_file = resources.files("chargax").joinpath(
        "data", "electricity_prices_kwh_2023_NL.csv"
    )
    with csv_file.open() as f:
        rows = list(csv.reader(f))[1:]
    return float(rows[(date.timetuple().tm_yday - 1) % len(rows)][1 + date.hour])


def _state_at(env: Chargax, date: datetime.datetime):
    _, state = env.reset(jax.random.PRNGKey(0))
    midnight = jdt.to_datetime(date.strftime("%Y-%m-%d"))
    return state._replace(
        datetime=midnight + jdt.Timedelta(seconds=date.hour * 3600 + date.minute * 60)
    )


@pytest.mark.parametrize(
    "date",
    [
        datetime.datetime(2024, 2, 28, 18, tzinfo=datetime.UTC),
        datetime.datetime(2024, 2, 29, 18, tzinfo=datetime.UTC),
        datetime.datetime(2024, 3, 1, 18, tzinfo=datetime.UTC),
        datetime.datetime(
            2024, 12, 31, 18, tzinfo=datetime.UTC
        ),  # day 366: wraps to the first dataset day
    ],
)
def test_grid_price_in_leap_year_maps_by_day_of_year(date):
    env = Chargax(
        station=ChargingStation.init_default_station(), simulation_starting_year=2024
    )
    state = _state_at(env, date)
    assert env.get_grid_buy_price(state) == pytest.approx(_csv_price(date), abs=1e-4)


def test_price_lookahead_crosses_midnight():
    env = Chargax(station=ChargingStation.init_default_station())
    date = datetime.datetime(2023, 11, 12, 22, tzinfo=datetime.UTC)
    obs = env.get_observation(_state_at(env, date))
    expected = [
        _csv_price(date + datetime.timedelta(hours=h))
        for h in range(env.price_hour_lookahead)
    ]
    assert np.allclose(obs["future_buy_prices"], expected, atol=1e-4)
