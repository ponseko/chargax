from abc import abstractmethod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, PRNGKeyArray

from chargax import Chargax, EnvState
from chargax._station_layout import StationBattery


class ChargaxBaselineAgent(eqx.Module):
    def __init__(self, env: Chargax):
        self.env = env

    def _run_episode(self, key: PRNGKeyArray, **kwargs) -> tuple[Array, Array]:
        """Run a single episode and return an tuple of [rewards, profits] at each step."""

        def _scan_step_fn(carry, _):
            seed, state, obs = carry
            this_step_key, next_key = jax.random.split(seed)
            action = self.get_action(this_step_key, env_state=state, observation=obs)
            timestep, new_state = self.env.step(this_step_key, state, action)
            obs, reward, *_ = timestep

            # Handle nested env_state (e.g. LogWrapper)
            env_state = getattr(new_state, "env_state", new_state)
            profit = env_state.profit

            return (next_key, new_state, obs), (reward, profit)

        obs, env_state = self.env.reset(key)
        _, (rewards, profits) = jax.lax.scan(
            _scan_step_fn,
            (key, env_state, obs),
            None,
            length=self.env.max_episode_steps,
        )
        return rewards, profits

    def evaluate(self, key: PRNGKeyArray, num_eval_episodes: int, **kwargs) -> float:
        """Evaluate the agent over multiple episodes and return the reward and profit as a stack."""
        keys = jax.random.split(key, num_eval_episodes)
        rewards = []
        profits = []
        for k in keys:
            r, p = self._run_episode(k)
            rewards.append(r)
            profits.append(p)

        # Average over the episodes
        avg_reward = np.stack(rewards)
        avg_profit = np.stack(profits)

        return avg_reward, avg_profit

    @abstractmethod
    def get_action(self, key: PRNGKeyArray, **kwargs) -> float:
        pass


class SimpleBatterySchedule(eqx.Module):
    """A simple rule-based battery schedule with hysteresis.
    Charges the battery when it drops below charge_threshold,
    discharges to support EVSEs once it reaches discharge_threshold."""

    is_charging: Bool[Array, "..."]  # Track charging state for each battery
    max_action: int = eqx.field(static=True)
    charge_threshold: float = eqx.field(static=True, default=0.05)
    discharge_threshold: float = eqx.field(static=True, default=0.4)

    def __call__(self, env_state: EnvState) -> tuple[Array, "SimpleBatterySchedule"]:
        batteries = env_state.grid.batteries_flat
        is_charging = jnp.array(self.is_charging)
        battery_percentages = batteries.battery_now / (batteries.capacity_kw + 1e-8)

        new_is_charging = jnp.where(
            battery_percentages >= self.discharge_threshold,
            jnp.bool_(False),
            jnp.where(
                battery_percentages <= self.charge_threshold,
                jnp.bool_(True),
                is_charging,
            ),
        )

        # Calc max charge action result:
        evses = env_state.grid.evses_flat
        max_current_per_port = jnp.minimum(
            evses.max_current, evses.car_max_current_intake
        )
        max_power_per_port_kw = evses.voltage * max_current_per_port / 1000.0
        total_demand_kw = jnp.sum(
            max_power_per_port_kw * evses.charger_is_car_connected
        )

        # Get the desired output of the batteries that are discharging:
        total_battery_throughput = jnp.sum(
            batteries.max_kw_throughput * new_is_charging
        )
        battery_discharge_fraction = jnp.clip(
            total_demand_kw / (total_battery_throughput + 1e-8), 0.0, 1.0
        )

        num_disc = self.max_action // 2
        discharge_action = jnp.round(
            num_disc * (1.0 - battery_discharge_fraction)
        ).astype(jnp.int32)

        action = jnp.where(
            new_is_charging,
            self.max_action,  # full charge
            discharge_action,  # proportional discharge matching car demand
        )

        new_self = eqx.tree_at(lambda s: s.is_charging, self, new_is_charging)
        return action, new_self


class FixedRateCharge(ChargaxBaselineAgent):
    """A rule-based baselinethat always selects a fixed, configurable charge
    rate for EVSEs (always positive / charging, never discharging, regardless
    of whether the environment allows discharging) and uses a schedule for
    station batteries."""

    env: Chargax
    charge_rate: float
    battery_schedule: SimpleBatterySchedule | None

    def __init__(
        self,
        env: Chargax,
        charge_rate: float = 0.5,
        battery_schedule: Literal["simple", "none"] = "simple",
    ):
        """
        charge_rate: fraction in [0, 1] of the maximum *charging* rate to apply
            at every EVSE, every timestep. 0.0 = idle, 1.0 = max charge.
            Always resolves to a non-negative (charging) action level, even if
            env.allow_discharging is True.
        """
        if not (0.0 <= charge_rate <= 1.0):
            raise ValueError(f"charge_rate must be in [0, 1], got {charge_rate}")

        self.env = env
        self.charge_rate = charge_rate

        if battery_schedule == "simple":
            is_charging_init = jax.tree.map(
                lambda _: False,
                self.env.station.batteries,
                is_leaf=lambda x: isinstance(x, StationBattery),
            )
            battery_schedule = SimpleBatterySchedule(
                max_action=env.num_discretization_levels * 2,
                is_charging=jnp.array(is_charging_init),
            )
        elif battery_schedule == "none":
            battery_schedule = None
        else:
            raise ValueError(f"Invalid battery_schedule: {battery_schedule}")
        self.battery_schedule = battery_schedule

    def _fixed_evse_action_level(self) -> int:
        """Compute the discretized action level corresponding to charge_rate."""
        num_levels = self.env.num_discretization_levels
        if self.env.allow_discharging:
            idle_level = num_levels
            max_level = num_levels * 2
        else:
            idle_level = 0
            max_level = num_levels

        level = idle_level + jnp.round(self.charge_rate * (max_level - idle_level))
        return jnp.clip(level, idle_level, max_level).astype(jnp.int32)

    def get_action(
        self, key: PRNGKeyArray, **kwargs
    ) -> tuple[dict, SimpleBatterySchedule | None]:
        action = self.env.sample_action(key)

        FIXED_ACTION_EVSES = self._fixed_evse_action_level()

        action["evses"] = jax.tree.map(
            lambda x: jnp.full_like(x, FIXED_ACTION_EVSES), action["evses"]
        )

        if self.battery_schedule is not None:
            env_state = kwargs.get("env_state")
            battery_action, new_schedule = self.battery_schedule(env_state)
            battery_action = [jnp.asarray(x) for x in battery_action]
            action["batteries"] = jax.tree.map(
                lambda x, y: jnp.full_like(x, y), action["batteries"], battery_action
            )
            return action, new_schedule
        else:
            IDLE = self.env.num_discretization_levels
            action["batteries"] = jax.tree.map(
                lambda x: jnp.full_like(x, IDLE), action["batteries"]
            )
            return action, None

    def _run_episode(self, key: PRNGKeyArray, **kwargs) -> tuple[Array, Array]:
        def _scan_step_fn(carry, _):
            seed, state, obs, _schedule = carry
            this_step_key, next_key = jax.random.split(seed)

            env_state = getattr(state, "env_state", state)
            action, new_schedule = self.get_action(
                this_step_key, env_state=env_state, observation=obs
            )
            timestep, new_state = self.env.step(this_step_key, state, action)
            obs, reward, *_ = timestep

            env_state = getattr(new_state, "env_state", new_state)
            return (next_key, new_state, obs, new_schedule), (
                reward,
                env_state.profit,
            )

        obs, env_state = self.env.reset(key)
        _, (rewards, profits) = jax.lax.scan(
            _scan_step_fn,
            (key, env_state, obs, self.battery_schedule),
            None,
            length=self.env.max_episode_steps,
        )
        return rewards, profits


class MaxCharge(FixedRateCharge):
    """A rule-based baseline based on FixedRateCharge that always selects the maximum charge
    rate for EVSEs."""

    def __init__(
        self,
        env: Chargax,
        battery_schedule: Literal["simple", "none"] = "simple",
    ):
        super().__init__(env, charge_rate=1.0, battery_schedule=battery_schedule)


class Random(ChargaxBaselineAgent):
    """A rule-based baseline that selects a random charge level."""

    env: Chargax
    state: None = None  # Placeholder for compatibility

    def get_action(self, key: PRNGKeyArray, **kwargs) -> float:
        return self.env.sample_action(key)
