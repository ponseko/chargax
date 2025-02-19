import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, NamedTuple
import equinox as eqx
import numpy as np
import time
from abc import abstractmethod

@chex.dataclass(frozen=True)
class EnvState:
    time: int


class TimeStep(NamedTuple):
    observation: chex.Array
    reward: Union[float, chex.Array]
    terminated: bool
    truncated: bool
    info: dict


class JaxBaseEnv(eqx.Module):
    """
    Base class for a JAX environment.
    This class inherits from eqx.Module, meaning it is a PyTree node and a dataclass.
    set params by setting the properties of the class.
    Much of the modules are inspired by the Gymnax base class.
    """

    # example_property: int = 0

    def __check_init__(self):
        """
        An equinox module that always runs on initialization.
        Can be used to check if parameters are set correctly, without overwriting __init__.
        """
        pass

    def step(
        self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array]
    ) -> Tuple[TimeStep, EnvState]:
        """Performs step transitions in the environment."""

        (obs_step, reward, terminated, truncated, info), state_step = self.step_env(
            key, state, action
        )
        
        obs_reset, state_reset = self.reset_env(key)

        done = jnp.any(jnp.logical_or(terminated, truncated))
        
        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )

        obs = jax.lax.select(done, obs_reset, obs_step)

        info["terminal_observation"] = obs_step

        return TimeStep(obs, reward, terminated, truncated, info), state

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key)
        return obs, state

    @abstractmethod
    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset transition."""
        raise NotImplementedError()

    @abstractmethod
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array]
    ) -> Tuple[TimeStep, EnvState]:
        """Environment-specific step transition."""
        raise NotImplementedError() 


class JaxEnvWrapper(eqx.Module):
    _env: eqx.Module 
    
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    returned_episode_returns: float
    train_timestep: int


class LogWrapper(JaxEnvWrapper):
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=0.,
            returned_episode_returns=0.,
            train_timestep=0,
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float, chex.Array],
    ) -> Tuple[TimeStep, LogEnvState]:
        (obs, reward, terminated, truncated, info), env_state = self._env.step(
            key, state.env_state, action
        )
        done = jnp.logical_or(terminated, truncated)
        new_episode_return = state.episode_returns + reward
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            train_timestep=state.train_timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode"] = done
        info["train_timestep"] = state.train_timestep
        return TimeStep(obs, reward, terminated, truncated, info), state
    




@chex.dataclass(frozen=True)
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: EnvState


class NormalizeVecObservation(JaxEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key: chex.PRNGKey):
        obs, state = self._env.reset(key)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key: chex.PRNGKey, state: NormalizeVecObsEnvState, action):
        timestep, env_state = self._env.step(
            key, state.env_state, action
        )
        obs = timestep.observation

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return TimeStep(
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            timestep.reward,
            timestep.terminated,
            timestep.truncated,
            timestep.info,
        ), state

