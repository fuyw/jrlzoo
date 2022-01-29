from typing import Any, Callable, Optional
import d4rl
import functools
import gym
import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
import time
from flax.core import FrozenDict
from flax.training import train_state
from models import Actor, DoubleCritic, Scalar, DynamicsModel
from utils import InfoBuffer, ReplayBuffer
from tqdm import trange


class OracleAgent:
    def __init__(self,
                 env: str = "hopper-medium-v2",
                 obs_dim: int = 11,
                 act_dim: int = 3,
                 seed: int = 42,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 lr: float = 3e-5,
                 lr_actor: float = 3e-4,
                 auto_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = False,
                 num_random: int = 10,
                 with_lagrange: bool = False,
                 lagrange_thresh: int = 5.0,

                 # COMBO
                 horizon: int = 5,
                 lr_model: float = 1e-3,
                 weight_decay: float = 5e-5,
                 real_ratio: float = 0.5,
                 ensemble_num: int = 7,
                 elite_num: int = 5,
                 max_patience: int = 5,
                 batch_size: int = 256,
                 rollout_batch_size: int = 10000,
                 holdout_ratio: float = 0.1,
                 model_dir: str = 'ensemble_models'):

        self.env = gym.make(env)
        self.update_step = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.lr_actor = lr_actor
        self.auto_entropy_tuning = auto_entropy_tuning
        self.backup_entropy = backup_entropy
        if target_entropy is None:
            self.target_entropy = -self.act_dim
        else:
            self.target_entropy = target_entropy

        # COMBO parameters
        self.horizon = horizon
        self.lr_model = lr_model
        self.weight_decay = weight_decay
        self.max_patience = max_patience
        self.holdout_ratio = holdout_ratio
        self.ensemble_num = ensemble_num
        self.real_ratio = real_ratio 

        # Initialize random keys
        self.rng = jax.random.PRNGKey(seed)
        self.rng, self.rollout_rng = jax.random.split(self.rng, 2)
        actor_key, critic_key, model_key = jax.random.split(self.rng, 3)

        # Dummy inputs
        dummy_obs = jnp.ones([1, self.obs_dim], dtype=jnp.float32)
        dummy_act = jnp.ones([1, self.act_dim], dtype=jnp.float32)
        dummy_model_inputs = jnp.ones([self.ensemble_num, self.obs_dim+self.act_dim], dtype=jnp.float32)

        # Initialize the Actor
        self.actor = Actor(self.act_dim)
        actor_params = self.actor.init(actor_key, actor_key, dummy_obs)["params"]
        self.actor_state = train_state.TrainState.create(
            apply_fn=Actor.apply,
            params=actor_params,
            tx=optax.adam(self.lr_actor))

        # Initialize the Critic
        self.critic = DoubleCritic()
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_act)["params"]
        self.critic_target_params = critic_params
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(self.lr))

        # Initialize the Dynamics Model
        self.model = DynamicsModel(env=env, seed=seed,
                                   ensemble_num=ensemble_num,   
                                   elite_num=elite_num,
                                   model_dir=model_dir)

        # Entropy tuning
        if self.auto_entropy_tuning:
            self.rng, alpha_key = jax.random.split(self.rng, 2)
            self.log_alpha = Scalar(0.0)
            self.alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_alpha.init(alpha_key)["params"],
                tx=optax.adam(self.lr)
            )

        # CQL parameters
        self.num_random = num_random
        self.with_lagrange = with_lagrange
        self.min_q_weight = 3.0 if not with_lagrange else 1.0
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.rng, cql_key = jax.random.split(self.rng, 2)
            self.log_cql_alpha = Scalar(0.0)  # 1.0
            self.cql_alpha_state = train_state.TrainState.create(
                apply_fn=None,
                params=self.log_cql_alpha.init(cql_key)["params"],
                tx=optax.adam(self.lr))

        # replay buffer
        self.batch_size = batch_size
        self.rollout_batch_size = rollout_batch_size
        self.real_batch_size = int(real_ratio * batch_size)
        self.model_batch_size = batch_size - self.real_batch_size
        self.masks = np.concatenate([np.ones(self.real_batch_size), np.zeros(self.model_batch_size)])

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self,
                   observations: jnp.array,
                   actions: jnp.array,
                   rewards: jnp.array,
                   discounts: jnp.array,
                   next_observations: jnp.array,
                   critic_target_params: FrozenDict,
                   actor_state: train_state.TrainState,
                   critic_state: train_state.TrainState,
                   alpha_state: train_state.TrainState,
                   key: jnp.ndarray):

        # For use in loss_fn without apply gradients
        frozen_actor_params = actor_state.params
        frozen_critic_params = critic_state.params

        def loss_fn(actor_params: FrozenDict,
                    critic_params: FrozenDict,
                    alpha_params: FrozenDict,
                    observation: jnp.ndarray,
                    action: jnp.ndarray,
                    reward: jnp.ndarray,
                    discount: jnp.ndarray,
                    next_observation: jnp.ndarray,
                    mask: jnp.ndarray,
                    rng: jnp.ndarray):
            """compute loss for a single transition"""
            rng, rng1, rng2 = jax.random.split(rng, 3)

            # Sample actions with Actor
            _, sampled_action, logp = self.actor.apply({"params": actor_params}, rng1, observation)

            # Alpha loss: stop gradient to avoid affect Actor parameters
            log_alpha = self.log_alpha.apply({"params": alpha_params})
            alpha_loss = -log_alpha * jax.lax.stop_gradient(logp + self.target_entropy)
            alpha = jnp.exp(log_alpha)

            # We use frozen_params so that gradients can flow back to the actor without being used to update the critic.
            sampled_q1, sampled_q2 = self.critic.apply({"params": frozen_critic_params}, observation, sampled_action)
            sampled_q = jnp.squeeze(jnp.minimum(sampled_q1, sampled_q2))

            # Actor loss
            alpha = jax.lax.stop_gradient(alpha)  # stop gradient to avoid affect Alpha parameters
            actor_loss = (alpha * logp - sampled_q)

            # Critic loss
            q1, q2 = self.critic.apply({"params": critic_params}, observation, action)
            q1 = jnp.squeeze(q1)
            q2 = jnp.squeeze(q2)

            # Use frozen_actor_params to avoid affect Actor parameters
            _, next_action, logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng2, next_observation)
            next_q1, next_q2 = self.critic.apply({"params": critic_target_params}, next_observation, next_action)

            next_q = jnp.squeeze(jnp.minimum(next_q1, next_q2))
            if self.backup_entropy:
                next_q -= alpha * logp_next_action
            target_q = reward + self.gamma * discount * next_q
            critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2

            # COMBO CQL loss
            rng3, rng4 = jax.random.split(rng, 2)

            # sample random actions
            cql_random_actions = jax.random.uniform(rng3, shape=(self.num_random, self.act_dim), minval=-1.0, maxval=1.0)

            # repeat next observations
            repeat_next_observations = jnp.repeat(jnp.expand_dims(next_observation, axis=0), repeats=self.num_random, axis=0)
            cql_random_q1, cql_random_q2 = self.critic.apply({"params": critic_params}, repeat_next_observations, cql_random_actions)
            _, cql_next_actions, cql_logp_next_action = self.actor.apply({"params": frozen_actor_params}, rng4, repeat_next_observations)
            cql_next_q1, cql_next_q2 = self.critic.apply({"params": critic_params}, repeat_next_observations, cql_next_actions)

            random_density = np.log(0.5 ** self.act_dim)
            cql_concat_q1 = jnp.concatenate([
                jnp.squeeze(cql_random_q1) - random_density,
                jnp.squeeze(cql_next_q1) - cql_logp_next_action,
            ])
            cql_concat_q2 = jnp.concatenate([
                jnp.squeeze(cql_random_q2) - random_density,
                jnp.squeeze(cql_next_q2) - cql_logp_next_action,
            ])

            # compute logsumexp loss w.r.t model_states
            # TODO: Fix Here ==> tf.boolean_mask(tf.concat(...), mask)
            cql1_loss = (jax.scipy.special.logsumexp(cql_concat_q1)*(1-mask) - q1*mask/self.real_ratio) * self.min_q_weight
            cql2_loss = (jax.scipy.special.logsumexp(cql_concat_q2)*(1-mask) - q2*mask/self.real_ratio) * self.min_q_weight

            total_loss = 0.5*critic_loss + actor_loss + alpha_loss + cql1_loss + cql2_loss

            log_info = {"critic_loss": critic_loss, "actor_loss": actor_loss, "alpha_loss": alpha_loss,
                        "cql1_loss": cql1_loss, "cql2_loss": cql2_loss, "sampled_q": sampled_q.mean(), "target_q": target_q.mean(),
                        "q1": q1, "q2": q2,  "cql_next_q1": cql_next_q1.mean(), "cql_next_q2": cql_next_q2.mean(),
                        "random_q1": cql_random_q1.mean(), "random_q2": cql_random_q2.mean(),
                        "alpha": alpha, "logp": logp, "logp_next_action": logp_next_action,
                        "cql_logp_next_action": cql_logp_next_action.mean()}

            return total_loss, log_info

        grad_fn = jax.vmap(
            jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0))
        rng = jnp.stack(jax.random.split(key, num=actions.shape[0]))

        (_, log_info), gradients = grad_fn(actor_state.params,
                                           critic_state.params,
                                           alpha_state.params,
                                           observations,
                                           actions,
                                           rewards,
                                           discounts,
                                           next_observations,
                                           self.masks,
                                           rng)

        gradients = jax.tree_map(functools.partial(jnp.mean, axis=0), gradients)
        log_info = jax.tree_map(functools.partial(jnp.mean, axis=0), log_info)

        actor_grads, critic_grads, alpha_grads = gradients

        # Update TrainState
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        alpha_state = alpha_state.apply_gradients(grads=alpha_grads)

        return log_info, actor_state, critic_state, alpha_state

    @functools.partial(jax.jit, static_argnames=("self"))
    def update_target_params(self, params: FrozenDict, target_params: FrozenDict):
        def _update(param, target_param):
            return self.tau * param + (1 - self.tau) * target_param

        updated_params = jax.tree_multimap(_update, params, target_params)
        return updated_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def select_action(self, params: FrozenDict, rng: Any, observation: np.ndarray, eval_mode: bool = False) -> jnp.ndarray:
        observation = jax.device_put(observation[None])
        rng, sample_rng = jax.random.split(rng)
        mean_action, sampled_action, _ = self.actor.apply({"params": params}, sample_rng, observation)
        return rng, jnp.where(eval_mode, mean_action.flatten(), sampled_action.flatten())

    def update(self, replay_buffer, model_buffer):
        self.rollout_batch_size = 30
        if self.update_step % 1 == 0: 
            sampled_batch = replay_buffer.sample(self.rollout_batch_size)
            _ = self.env.reset()
            env_state = self.env.sim.get_state()
            for i in range(self.rollout_batch_size):
                # reset the env state
                env_state.qpos[:] = sampled_batch.qpos[i]
                env_state.qvel[:] = sampled_batch.qvel[i]
                self.env.sim.set_state(env_state)

                # rollout the model
                obs = sampled_batch.observations[i]
                step, done = 0, False
                while (not done) and (step < 5):
                    self.rollout_rng, action = self.select_action(
                        self.actor_state.params, self.rollout_rng, obs, True)
                    next_obs, reward, done, _ = self.env.step(action)
                    model_buffer.add(obs, action, next_obs, reward, done)

        # sample from real & model buffer
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = model_buffer.sample(self.model_batch_size)
        # model_batch = replay_buffer.sample(self.real_batch_size)

        concat_observations = np.concatenate([real_batch.observations, model_batch.observations], axis=0)
        concat_actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0)
        concat_rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0)
        concat_discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0)
        concat_next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        log_info, self.actor_state, self.critic_state, self.alpha_state = self.train_step(
            concat_observations, concat_actions, concat_rewards, concat_discounts,
            concat_next_observations, self.critic_target_params, self.actor_state,
            self.critic_state, self.alpha_state, key
        )
        log_info['batch_rewards'] = real_batch.rewards.mean().item()
        log_info['batch_discounts'] = real_batch.discounts.mean().item()
        log_info['batch_act'] = abs(real_batch.actions).sum(1).mean().item()
        log_info['model_rewards'] = model_batch.rewards.mean().item()
        log_info['model_discounts'] = model_batch.discounts.mean().item()
        log_info['model_act'] = abs(model_batch.actions).sum(1).mean().item()

        # upate target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(params, target_params)

        self.update_step += 1
        return log_info

    def update1(self, replay_buffer, model_buffer):
        real_batch = replay_buffer.sample(self.real_batch_size)
        model_batch = replay_buffer.sample(self.model_batch_size)

        concat_observations = np.concatenate([real_batch.observations, model_batch.observations], axis=0)
        concat_actions = np.concatenate([real_batch.actions, model_batch.actions], axis=0)
        concat_rewards = np.concatenate([real_batch.rewards, model_batch.rewards], axis=0)
        concat_discounts = np.concatenate([real_batch.discounts, model_batch.discounts], axis=0)
        concat_next_observations = np.concatenate([real_batch.next_observations, model_batch.next_observations], axis=0)

        # CQL training with COMBO
        self.rng, key = jax.random.split(self.rng)
        log_info, self.actor_state, self.critic_state, self.alpha_state = self.train_step(
            concat_observations, concat_actions, concat_rewards, concat_discounts,
            concat_next_observations, self.critic_target_params, self.actor_state,
            self.critic_state, self.alpha_state, key
        )
        log_info['batch_rewards'] = real_batch.rewards.mean().item()
        log_info['model_rewards'] = model_batch.rewards.mean().item()

        # upate target network
        params = self.critic_state.params
        target_params = self.critic_target_params
        self.critic_target_params = self.update_target_params(params, target_params)

        self.update_step += 1
        return log_info


def eval_policy(agent, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    eval_rng = jax.random.PRNGKey(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        t = 0
        obs, done = eval_env.reset(), False
        while not done:
            t += 1
            eval_rng, action = agent.select_action(agent.actor_state.params, eval_rng, np.array(obs), True)
            obs, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    return d4rl_score


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--task", default="Hopper")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr_actor", default=3e-5, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--eval_freq", default=int(5e3), type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--auto_entropy_tuning", default=True, action="store_false")
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--model_dir", default="./ensemble_models", type=str)
    parser.add_argument("--backup_entropy", default=False, action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    # Env parameters
    # env = gym.make(f"{args.task}-v2")
    env = gym.make(f"hopper-medium-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # random seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)

    # TD3 agent
    agent = OracleAgent(env=args.env, obs_dim=obs_dim, act_dim=act_dim, seed=args.seed,
                        lr=args.lr, lr_actor=args.lr_actor, rollout_batch_size=10000)

    fix_obs = np.random.normal(size=(128, obs_dim))
    fix_act = np.random.normal(size=(128, act_dim))

    # Replay buffer
    env_state = env.sim.get_state()
    qpos_dim = len(env_state.qpos)
    qvel_dim = len(env_state.qvel)

    # Replay buffer
    replay_buffer = InfoBuffer(obs_dim, act_dim, qpos_dim, qvel_dim)
    replay_buffer.load(f'saved_buffers/{args.task}-v2/s0.npz')

    # replay_buffer = ReplayBuffer(obs_dim, act_dim)
    # replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    model_buffer = ReplayBuffer(obs_dim, act_dim, max_size=int(5e5))

    # Evaluate the untrained policy
    logs = [{"step": 0, "reward": eval_policy(agent, args.env, args.seed)}]  # 2.382196339051178

    # Initialize training stats
    start_time = time.time()

    # Train agent and evaluate policy
    for t in trange(args.max_timesteps):
        log_info = agent.update1(replay_buffer, model_buffer)

        if (t + 1) % 5000 == 0:
            eval_reward = eval_policy(agent, args.env, args.seed)
            log_info.update({
                "step": t+1,
                "reward": eval_reward,
                "time": (time.time() - start_time) / 60
            })
            logs.append(log_info)
            fix_q1, fix_q2 = agent.critic.apply({"params": agent.critic_state.params}, fix_obs, fix_act)
            print(
                f"\n# Step {t+1}: eval_reward = {eval_reward:.2f}\n"
                f"\talpha_loss: {log_info['alpha_loss']:.2f}, alpha: {log_info['alpha']:.2f}, logp: {log_info['logp']:.2f}\n"
                f"\tactor_loss: {log_info['actor_loss']:.2f}, sampled_q: {log_info['sampled_q']:.2f}\n"
                f"\tcritic_loss: {log_info['critic_loss']:.2f}, q1: {log_info['q1']:.2f}, q2: {log_info['q2']:.2f}, target_q: {log_info['target_q']:.2f}\n"
                f"\tcql_next_q1: {log_info['cql_next_q1']:.2f}, random_q1: {log_info['random_q1']:.2f} \n"
                f"\tcql_next_q2: {log_info['cql_next_q2']:.2f}, random_q2: {log_info['random_q2']:.2f} \n"
                f"\tlogp_next_action: {log_info['logp_next_action']:.2f},  cql_logp_next_action: {log_info['cql_logp_next_action']:.2f}\n"
                f"\tcql1_loss: {log_info['cql1_loss']:.2f}, cql2_loss: {log_info['cql2_loss']:.2f}\n" 
                f"\tbatch_rewards: {log_info['batch_rewards']:.2f}, batch_discounts: {log_info['batch_discounts']:.2f}, batch_act: {log_info['batch_act']:.2f}\n"
                f"\tmodel_rewards: {log_info['model_rewards']:.2f}, model_discounts: {log_info['model_discounts']:.2f}, model_act: {log_info['model_act']:.2f}\n"
                f"\tfix_q1: {fix_q1.squeeze().mean().item():.2f}, fix_q2: {fix_q2.squeeze().mean().item():.2f}\n"
            )

    # Save logs
    log_name = f"s{args.seed}"
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{args.log_dir}/{args.env}/{log_name}.csv")
    # agent.save(f"{args.model_dir}/{args.env}/{args.seed}")


if __name__ == "__main__":
    args = get_args()
    print(f"\nArguments:\n{vars(args)}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.log_dir}/{args.env}", exist_ok=True)
    os.makedirs(f"{args.model_dir}/{args.env}", exist_ok=True)
    main(args)

