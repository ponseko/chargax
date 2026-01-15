from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from scipy.optimize import milp as milp

from chargax import Chargax, EnvState

from ._rulebased import MaxCharge


class LPAgent(eqx.Module):
    env: Chargax
    rollout_fn: Callable[..., EnvState]
    state: None = None  # Placeholder for compatibility
    rollout_agent: Any = MaxCharge
    oracle_mode: bool = True  # Not a true oracle

    def __post_init__(self):
        setattr(self, "rollout_agent", self.rollout_agent(self.env))

    @staticmethod
    def solve_milp(action_space, env_state, future_states):
        actions = jnp.ones_like(action_space)
        return (
            milp(
                c=1,
                # constraints=,
            ).x
            * actions
        ).astype(jnp.int32)

    def get_action(
        self, key: PRNGKeyArray, observation, env_state: EnvState, **kwargs
    ) -> jnp.ndarray:
        # NOTE: future_states includes states after the episode has technically ended
        # What is in them is a bit undefined for now ...
        future_states = self._simulate_rollout(key, observation, env_state).env_state

        output_shape = jnp.ones_like(self.env.action_space.nvec)
        optimization_result = jax.pure_callback(
            self.solve_milp,
            output_shape,
            self.env.action_space.nvec,
            env_state,
            future_states,
        )

        return optimization_result

    def _simulate_rollout(self, key: PRNGKeyArray, observation, state) -> jnp.ndarray:
        if not self.oracle_mode:  # different key from the real step
            _, key = jax.random.split(key)

        states = self.rollout_fn(self.rollout_agent, self.env, key, state, observation)

        return states
