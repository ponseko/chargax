import chex
import numpy as np
import jax.numpy as jnp
import jax 
from typing import Any, Union, Tuple

class Space:
    """Minimal jittable class for abstract gymnax space."""

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: int) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete gymnax spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int16

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: int) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class MultiDiscrete(Space):
    """
    Minimal implementation of a MultiDiscrete space.
    input nvec: array of integers representing the number of discrete values in each dimension
    """

    def __init__(self, nvec: chex.Array, dtype: jnp.dtype = jnp.int8, start: int = 0):
        assert (
            len(nvec.shape) == 1 and nvec.shape[0] > 0
        ), "nvec must be a 1D array with at least one element"
        self.nvec = nvec
        self.shape = nvec.shape
        self.n = self.shape[0]
        self.dtype = dtype
        self.start = start

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, self.shape, self.start, self.nvec)

    def contains(self, x: chex.Array) -> bool:
        return (
            x.shape == self.shape
            and jnp.all(x >= self.start)
            and jnp.all(x < self.nvec)
        )
    

class Box(Space):
    """Minimal jittable class for array-shaped gymnax spaces."""

    def __init__(
        self,
        low: Union[jnp.ndarray, float],
        high: Union[jnp.ndarray, float],
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: int) -> jnp.ndarray:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond