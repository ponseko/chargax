import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from functools import partial
import distrax

from chargax import Chargax, LogWrapper

@chex.dataclass(frozen=True)
class TrainerParams:
    rng: int = 42
    num_envs: int = 4
    total_timesteps: int = 10_000

def build_random_trainer(
        env: Chargax,
        params: TrainerParams = TrainerParams()
):
    env = env
    env = LogWrapper(env)
    params = params
    rng = jax.random.PRNGKey(params.rng)

    rng, reset_key = jax.random.split(rng)

    # obs_v, env_state_v = env.reset(reset_key) #
    obs_v, env_state_v = jax.vmap(env.reset)(jax.random.split(reset_key, params.num_envs))


    def train_function(rng: chex.PRNGKey = rng):

        def env_step(runner_state, _):
            # rng, obs_v, env_state_v = runner_state
            # action_key, step_key = jax.random.split(rng, 2)
            # action_v = env.action_space.sample(action_key)
            # (obs_v, reward, terminated, truncated, info), env_state_v = env.step(
            #     step_key, env_state_v, action_v
            # )
            # return (rng, obs_v, env_state_v), None
            rng, obs_v, env_state_v = runner_state

            rng, action_key, step_key = jax.random.split(rng, 3)
            action_v = jax.vmap(
                env.action_space.sample
            )(jax.random.split(action_key, params.num_envs))

            (obs_v, reward, terminated, truncated, info), env_state_v = jax.vmap(
                env.step
            )(
                jax.random.split(step_key, params.num_envs),
                env_state_v,
                action_v
            )

            return (rng, obs_v, env_state_v), info

        initial_runner_state = (rng, obs_v, env_state_v)
        trained_runner_state, train_metrics = jax.lax.scan(
            env_step,
            initial_runner_state,
            None,
            length=params.total_timesteps // params.num_envs,
        )

        return trained_runner_state, train_metrics

    return train_function