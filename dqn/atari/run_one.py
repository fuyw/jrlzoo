import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state
from tqdm import trange

from atari_wrappers import wrap_deepmind
from utils import Experience, ReplayBuffer, get_logger, linear_schedule


###################
# Utils Functions #
###################
def eval_policy(apply_fn, state, env):
    t1 = time.time()
    obs = env.reset()
    act_counts = np.zeros(env.action_space.n)
    while not env.get_real_done():
        action = sample(apply_fn, state.params, np.moveaxis(obs[None], 1, -1))
        act_counts[action] += 1
        obs, _, done, _ = env.step(action.item())
        if done:
            obs = env.reset()
    act_counts /= act_counts.sum()
    act_counts = ", ".join([f"{i:.2f}" for i in act_counts])
    return np.mean(env.get_eval_rewards()), act_counts, time.time() - t1


#############
# DQN Agent #
#############
init_fn = nn.initializers.xavier_uniform()
class QNetwork(nn.Module):
    act_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1")
        self.conv2 = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2")
        self.conv3 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3")
        self.fc_layer = nn.Dense(features=512, name="fc")
        self.out_layer = nn.Dense(features=self.act_dim, name="out")

    def __call__(self, observation):
        x = observation.astype(jnp.float32) / 255.  # (84, 84, 4)
        x = nn.relu(self.conv1(x))                  # (21, 21, 32)
        x = nn.relu(self.conv2(x))                  # (11, 11, 64)
        x = nn.relu(self.conv3(x))                  # (11, 11, 64)
        x = x.reshape(len(observation), -1)         # (7744,)
        x = nn.relu(self.fc_layer(x))               # (512,)
        Qs = self.out_layer(x)                      # (act_dim,)
        return Qs


@partial(jax.jit, static_argnums=0)
def sample(apply_fn, params, observation):
    """sample action s ~ pi(a|s)"""
    logits = apply_fn({"params": params}, observation)
    action = logits.argmax(1)
    return action


#################
# Main Function #
#################
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-name", type=str, default="Breakout")
    parser.add_argument("--buffer-size", type=int, default=int(1e6))
    parser.add_argument("--total-timesteps", type=int, default=int(5e6))
    parser.add_argument("--warmup-timesteps", type=int, default=int(5e4))
    parser.add_argument("--eval-num", type=int, default=100)
    parser.add_argument("--ckpt-num", type=int, default=10)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--explore_freq", type=float, default=0.1)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    return args
args = get_args()


def run(args):
    # general setting
    start_time = time.time()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f"dqn_s{args.seed}_{timestamp}"
    exp_info = f'# Running experiment for: {exp_name}_{args.env_name} #'
    eval_freq = args.total_timesteps // args.eval_num
    ckpt_freq = args.total_timesteps // args.ckpt_num
    print('#' * len(exp_info) + f'\n{exp_info}\n' + '#' * len(exp_info))
    logger = get_logger(f"logs/{exp_name}.log")

    # make environments
    env = gym.make(f"{args.env_name}NoFrameskip-v4")
    env = wrap_deepmind(env, dim=84, framestack=False, obs_format="NCHW")
    eval_env = gym.make(f"{args.env_name}NoFrameskip-v4")
    eval_env = wrap_deepmind(eval_env, dim=84, obs_format="NCHW", test=True)
    act_dim = env.action_space.n

    # initialize DQN agent
    rng = jax.random.PRNGKey(args.seed)
    q_network = QNetwork(act_dim)
    params = q_network.init(rng, jnp.ones(shape=(1, 84, 84, 4)))["params"]
    lr_scheduler = optax.linear_schedule(init_value=args.lr, end_value=1e-6,
                                         transition_steps=args.total_timesteps)
    state = train_state.TrainState.create(apply_fn=q_network.apply, params=params,
                                          tx=optax.adam(lr_scheduler))
    target_params = params

    # create the replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # update function
    @jax.jit
    def update_jit(state, target_params, batch):
        next_Q = q_network.apply({"params": target_params}, batch.next_observations).max(-1)
        target_Q = batch.rewards + args.gamma * next_Q * batch.discounts
        def loss_fn(params):
            Qs = q_network.apply({"params": params}, batch.observations)
            Q = jax.vmap(lambda q,a: q[a])(Qs, batch.actions.reshape(-1, 1)).squeeze()
            loss = (Q - target_Q) ** 2
            log_info = {
                "avg_Q": Q.mean(),
                "min_Q": Q.min(),
                "max_Q": Q.max(),
                "avg_target_Q": target_Q.mean(),
                "min_target_Q": target_Q.min(),
                "max_target_Q": target_Q.max(),
                "avg_loss": loss.mean(),
                "max_loss": loss.max(),
                "min_loss": loss.min(),
            }
            return loss.mean(), log_info
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, log_info

    # start training
    obs = env.reset()   # (84, 84)
    for t in trange(1, 1+args.total_timesteps):
        # select action
        epsilon = linear_schedule(1.0, 0.05, args.explore_freq*args.total_timesteps, t)
        if t <= args.warmup_timesteps:
            action = np.random.choice(act_dim)
        else:
            if np.random.random() < epsilon:
                action = np.random.choice(act_dim)
            else:
                context = replay_buffer.recent_obs()
                context.append(obs)
                context = np.stack(context, axis=-1)[None]
                action = sample(state.apply_fn, state.params, context).item()
        
        # interact with the environment
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(Experience(obs, action, reward, done))
        obs = next_obs
        if done:
            obs = env.reset()
        
        # update the agent
        if (t > args.warmup_timesteps) and (t % args.train_freq == 0):
            batch = replay_buffer.sample_batch(args.batch_size)
            state, log_info = update_jit(state, target_params, batch)
            if t % args.target_update_freq == 0:
                target_params = state.params
        
        # evaluate the agent
        if (t > args.warmup_timesteps) and (t % eval_freq == 0):
            eval_reward, act_counts, eval_time = eval_policy(q_network.apply, state, eval_env)
            print(f"Eval at {t//1000}K: reward = {eval_reward:.1f}, eval_time={eval_time:.1f}s, total_time={(time.time()-start_time)/60:.1f}min\n"
                  f"\tavg_loss: {log_info['avg_loss']:.3f}, max_loss: {log_info['max_loss']:.3f}, "
                  f"min_loss: {log_info['min_loss']:.3f}\n"
                  f"\tavg_Q: {log_info['avg_Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                  f"min_Q: {log_info['min_Q']:.3f}\n"
                  f"\tavg_target_Q: {log_info['avg_target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                  f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                  f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, max_batch_rewards: {batch.rewards.max():.3f}, "
                  f"min_batch_rewards: {batch.rewards.min():.3f}\n"
                  f"\tavg_batch_discounts: {batch.discounts.mean():.3f}, act_counts: ({act_counts})\n"
                  f"\tlr={lr_scheduler(state.opt_state[1].count):.6f}, epsilon={epsilon:.6f}\n")
            logger.info(f"Eval at {t//1000}K: reward = {eval_reward:.1f}, eval_time={eval_time:.1f}s, total_time={(time.time()-start_time)/60:.1f}min\n"
                        f"\tavg_loss: {log_info['avg_loss']:.3f}, max_loss: {log_info['max_loss']:.3f}, "
                        f"min_loss: {log_info['min_loss']:.3f}\n"
                        f"\tavg_Q: {log_info['avg_Q']:.3f}, max_Q: {log_info['max_Q']:.3f}, "
                        f"min_Q: {log_info['min_Q']:.3f}\n"
                        f"\tavg_target_Q: {log_info['avg_target_Q']:.3f}, max_target_Q: {log_info['max_target_Q']:.3f}, "
                        f"min_target_Q: {log_info['min_target_Q']:.3f}\n"
                        f"\tavg_batch_rewards: {batch.rewards.mean():.3f}, max_batch_rewards: {batch.rewards.max():.3f}, "
                        f"min_batch_rewards: {batch.rewards.min():.3f}\n"
                        f"\tavg_batch_discounts: {batch.discounts.mean():.3f}, act_counts: ({act_counts})\n"
                        f"\tlr={lr_scheduler(state.opt_state[1].count):.6f}, epsilon={epsilon:.6f}\n")

        # save checkpoints
        if t % ckpt_freq == 0:
            checkpoints.save_checkpoint("ckpts", state, t//ckpt_freq, prefix="dqn_breakout", keep=20, overwrite=True)


if __name__ == "__main__":
    args = get_args()
    os.makedirs("logs", exist_ok=True)
    os.makedirs("ckpts", exist_ok=True)
    run(args)
