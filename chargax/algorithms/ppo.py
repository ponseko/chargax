import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import optax
from typing import NamedTuple, List
from dataclasses import replace

from chargax import Chargax, LogWrapper, NormalizeVecObservation
from chargax.algorithms.networks import ActorNetworkMultiDiscrete, CriticNetwork
import wandb

def create_ppo_networks(
    key,
    in_shape: int,
    actor_features: List[int],
    critic_features: List[int],
    actions_nvec: int,
):
    """Create PPO networks (actor critic)"""
    actor_key, critic1_key = jax.random.split(key)
    actor = ActorNetworkMultiDiscrete(actor_key, in_shape, actor_features, actions_nvec)
    critic = CriticNetwork(critic1_key, in_shape, critic_features)
    return actor, critic

@chex.dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 100.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    total_timesteps: int = 5e6
    num_envs: int = 12
    num_steps: int = 300 # steps per environment
    num_minibatches: int = 4 # Number of mini-batches
    update_epochs: int = 4 # K epochs to update the policy

    seed: int = 42
    debug: bool = False
    evaluate_deterministically: bool = False

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs
    
    @property
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches
    

# Define a simple tuple to hold the state of the environment. 
# This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array

class TrainState(NamedTuple):
    actor: eqx.Module
    critic: eqx.Module
    optimizer_state: optax.OptState

# Jit the returned function, not this function
def build_ppo_trainer(
        env: Chargax,
        config_params: dict = {},
        baselines: dict = {}, # Will be inserted every wandb log step
    ):

    # setup env (wrappers) and config
    env = LogWrapper(env)
    env = NormalizeVecObservation(env)
    # env = FlattenObservationWrapper(env)
    observation_space = env.observation_space
    action_space = env.action_space
    num_actions = action_space.n
    logging_baselines = baselines

    config = PPOConfig(**config_params)

    # rng keys
    rng = jax.random.PRNGKey(config.seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)

    # networks
    actor, critic = create_ppo_networks(
        key=network_key, 
        in_shape=observation_space.shape[0],
        actor_features=[256, 256], 
        critic_features=[256, 256], 
        actions_nvec=action_space.nvec
    )

    # optimizer
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_iterations
        )
        return config.learning_rate * frac
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if config.anneal_lr else config.learning_rate,
            eps=1e-5
        ),
    )
    optimizer_state = optimizer.init({
        "actor": actor,
        "critic": critic
    })

    train_state = TrainState(
        actor=actor,
        critic=critic,
        optimizer_state=optimizer_state
    )

    rng, key = jax.random.split(rng)
    reset_key = jax.random.split(key, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_key)

    def eval_func(train_state, rng):

        def step_env(carry):
            rng, obs, env_state, done, episode_reward = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            action_dist = train_state.actor(obs)
            if config.evaluate_deterministically:
                action = jnp.argmax(action_dist.logits, axis=-1)
            else:
                action = action_dist.sample(seed=action_key)
            (obs, reward, terminated, truncated, info), env_state = env.step(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)
            episode_reward += reward
            return (rng, obs, env_state, done, episode_reward)
        
        def cond_func(carry):
            _, _, _, done, _ = carry
            return jnp.logical_not(done)
        
        rng, reset_key = jax.random.split(rng)
        obs, env_state = env.reset(reset_key)
        done = False
        episode_reward = 0.0

        rng, obs, env_state, done, episode_reward = jax.lax.while_loop(cond_func, step_env, (rng, obs, env_state, done, episode_reward))

        return episode_reward

    def train_func(rng=rng):
        
        # functions prepended with _ are called in jax.lax.scan of train_step

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            action_dist = jax.vmap(train_state.actor)(last_obs)
            value = jax.vmap(train_state.critic)(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)

            # take a step in the environment
            rng, key = jax.random.split(rng)
            step_key = jax.random.split(key, config.num_envs)
            (obsv, reward, terminated, truncated, info), env_state= jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_key, env_state, action)
            done = jnp.logical_or(terminated, truncated)

            # jax.debug.breakpoint()

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info
            )

            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        def _calculate_gae(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            value, reward, done = (
                transition.value,
                transition.reward,
                transition.done,
            )
            delta = reward + config.gamma * next_value * (1 - done) - value
            gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
            return (gae, value), (gae, gae + value)
        
        def _update_epoch(update_state, _):
            """ Do one epoch of update"""

            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_los_fn(params, trajectory_minibatch, advantages, returns):
                action_dist = jax.vmap(params["actor"])(trajectory_minibatch.observation)
                log_prob = action_dist.log_prob(trajectory_minibatch.action).sum(axis=-1)
                entropy = action_dist.entropy().mean()
                value = jax.vmap(params["critic"])(trajectory_minibatch.observation)

                def ___ppo_actor_los():
                    # actor loss 
                    ratio = jnp.exp(log_prob - trajectory_minibatch.log_prob.sum(axis=-1))
                    _advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    actor_loss1 = _advantages * ratio
                    actor_loss2 = (
                        jnp.clip(
                            ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                        ) * _advantages
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    return actor_loss

                actor_loss = ___ppo_actor_los() 

                value_pred_clipped = trajectory_minibatch.value + (
                    jnp.clip(
                        value - trajectory_minibatch.value, -config.clip_coef_vf, config.clip_coef_vf
                    )
                )
                value_losses = jnp.square(value - returns)
                value_losses_clipped = jnp.square(value_pred_clipped - returns)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                # Total loss
                total_loss = (
                    actor_loss 
                    + config.vf_coef * value_loss
                    - config.ent_coef * entropy
                )
                return total_loss, (actor_loss, value_loss, entropy)
            
            def __update_over_minibatch(train_state: TrainState, minibatch):
                trajectory_mb, advantages_mb, returns_mb = minibatch
                (total_loss, _), grads = __ppo_los_fn({
                        "actor": train_state.actor,
                        "critic": train_state.critic
                    }, trajectory_mb, advantages_mb, returns_mb
                )
                updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state)
                new_networks = optax.apply_updates({
                    "actor": train_state.actor,
                    "critic": train_state.critic
                }, updates)
                train_state = TrainState(
                    actor=new_networks["actor"],
                    critic=new_networks["critic"],
                    optimizer_state=optimizer_state
                )
                return train_state, total_loss
            
            train_state, trajectory_batch, advantages, returns, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.batch_size)
            batch = (trajectory_batch, advantages, returns)
            
            # reshape (flatten over first dimension)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
            )
            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            # split in minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
            )
            # update over minibatches
            train_state, total_loss = jax.lax.scan(
                __update_over_minibatch, train_state, minibatches
            )
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            return update_state, total_loss

        def train_step(runner_state, _):

            # Do rollout of single trajactory
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # calculate gae
            train_state, env_state, last_obs, rng = runner_state
            last_value = jax.vmap(train_state.critic)(last_obs)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
    
            # Do update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state[0]
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            rng = update_state[-1]

            rng, eval_key = jax.random.split(rng)
            eval_rewards = eval_func(train_state, eval_key)
            metric["eval_rewards"] = eval_rewards

            def callback(info):
                if config.debug:
                    print(f'timestep={(info["train_timestep"][-1][0] * config.num_envs)}, eval rewards={info["eval_rewards"]}')
                if wandb.run:
                    if "logging_data" not in info:
                        info["logging_data"] = {}
                    finished_episodes = info["returned_episode"] 
                    if finished_episodes.any():
                        info["logging_data"] = jax.tree.map(
                            lambda x: x[finished_episodes].mean(), info["logging_data"]
                        )
                        wandb.log({
                            "timestep": info["train_timestep"][-1][0] * config.num_envs, 
                            "eval_rewards": info["eval_rewards"],
                            **info["logging_data"],
                            **logging_baselines
                        })

            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, _ 

        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, key)
        trained_runner_state, train_metrics = jax.lax.scan(
            train_step, runner_state, None, config.num_iterations
        )

        return trained_runner_state, train_metrics

    return train_func, config
