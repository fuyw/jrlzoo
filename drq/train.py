import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import ml_collections
import gym
import time
import random
import numpy as np

from tqdm import trange
from models import DrQAgent
from utils import get_logger, make_env, EfficientBuffer


PLANET_ACTION_REPEAT = {
    "cartpole-swingup": 8,
    "reacher-easy": 4,
    "cheetah-run": 4,
    "finger-spin": 2,
    "ball_in_cup-catch": 4,
    "walker-walk": 2
}


def eval_policy(agent: DrQAgent,
                env: gym.Env,
                eval_episodes: int = 10):
    t1 = time.time()
    avg_reward, avg_step = 0., 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            avg_step += 1
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    return avg_reward, avg_step, time.time() - t1


def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"drq_s{config.seed}_{timestamp}"
    exp_info = f"# Running experiment for: {exp_name}_{config.env_name} #"
    print("#"*len(exp_info) + f"\n{exp_info}\n" + "#"*len(exp_info))

    logger = get_logger(f"logs/{config.env_name.lower()}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # set random seed
    np.random.seed(config.seed)
    random.seed(config.seed)

    # initialize environments
    action_repeat = PLANET_ACTION_REPEAT.get(config.env_name, 2)
    env = make_env(env_name=config.env_name,
                   seed=config.seed,
                   action_repeat=action_repeat,
                   num_stack=config.num_stack,
                   image_size=config.image_size,
                   from_pixels=True)
    eval_env = make_env(env_name=config.env_name,
                        seed=config.seed+42,
                        action_repeat=action_repeat,
                        num_stack=config.num_stack,
                        image_size=config.image_size,
                        from_pixels=True)

    # DrQAgent
    obs_shape = env.observation_space["pixels"].shape
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = DrQAgent(obs_shape=obs_shape,
                     act_dim=act_dim,
                     max_action=max_action,
                     emb_dim=config.emb_dim,
                     seed=config.seed,
                     lr=config.lr,
                     tau=config.tau,
                     gamma=config.gamma,
                     hidden_dims=config.hidden_dims,
                     cnn_features=config.cnn_features,
                     cnn_kernels=config.cnn_kernels,
                     cnn_strides=config.cnn_strides,
                     cnn_padding=config.cnn_padding)

    # replay buffer
    buffer_size = config.max_timesteps // action_repeat
    replay_buffer = EfficientBuffer(obs_shape=obs_shape,
                                    act_dim=act_dim,
                                    batch_size=config.batch_size,
                                    max_size=buffer_size)
    replay_buffer_iterator = replay_buffer.get_iterator()
    logs = [{
        "step": 0,
        "reward": eval_policy(agent, eval_env, config.eval_episodes)[0]
    }]

    # start training
    observation, done = env.reset(), False
    for t in trange(action_repeat, config.max_timesteps+action_repeat, action_repeat):
        if t < config.start_timesteps * action_repeat:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(observation)

        next_observation, reward, done, info = env.step(action)
        done_bool = int(done) if "TimeLimit.truncated" not in info else 0

        replay_buffer.add(observation["pixels"],
                          action,
                          next_observation["pixels"],
                          reward,
                          done_bool,
                          done)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if t >= config.start_timesteps * action_repeat:
            batch = next(replay_buffer_iterator)
            log_info = agent.update(batch)

        if ((t>int(9.5e5) and (t % config.eval_freq == 0)) or (
                t<=int(9.5e5) and t % (2*config.eval_freq) == 0)):
            eval_reward, eval_step, eval_time = eval_policy(agent, eval_env, config.eval_episodes)

            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "reward": eval_reward,
                    "eval_time": eval_time,
                    "eval_step": eval_step,
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.3f}, "
                    f"eval_time: {eval_time:.0f}, "
                    f"time: {log_info['time']:.3f}\n"
                    f"\tactor_loss: {log_info['actor_loss']:.3f}, "
                    f"critic_loss: {log_info['critic_loss']:.3f}, "
                    f"alpha_loss: {log_info['alpha_loss']:.3f}\n"
                    f"\tq: {log_info['q']:.3f}, target_q: {log_info['target_q']:.3f}, "
                    f"logp: {log_info['logp']:.3f}, alpha: {log_info['alpha']:.3f}\n"
                    f"\tbatch_R: {batch.rewards.mean():.3f}, "
                    f"batch_Rmax: {batch.rewards.max():.3f}, "
                    f"batch_Rmin: {batch.rewards.min():.3f}\n"
                )
                logs.append(log_info)
            else:
                logs.append({"step": t, "reward": eval_reward})
                logger.info(
                    f"\n[#Step {t}] eval_reward: {eval_reward:.3f}, eval_time: {eval_time:.0f}\n"
                )

    # Save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        f"{config.log_dir}/{config.env_name}/{exp_name}.csv")
