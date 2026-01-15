import equinox as eqx
import numpy as np
from jaxtyping import PRNGKeyArray

from chargax import Chargax


class MaxCharge(eqx.Module):
    """A rule-based baseline that always selects the maximum charge level."""

    env: Chargax
    state: None = None  # Placeholder for compatibility

    def get_action(self, key: PRNGKeyArray, **kwargs) -> float:
        return self.env.action_space.sample(key)


class Random(eqx.Module):
    """A rule-based baseline that selects a random charge level."""

    env: Chargax
    state: None = None  # Placeholder for compatibility

    def get_action(self, key: PRNGKeyArray, **kwargs) -> float:
        MAX_ACTION = self.env.action_space.nvec.max()
        action = np.ones_like(self.env.action_space.nvec) * MAX_ACTION
        if self.env.include_battery:
            action[-1] = 0.0  # Maximum discharge for the battery
        return action
