import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
import time
from typing import Tuple

import gym
import ml_collections
import numpy as np
import pandas as pd
from tqdm import trange


from models import PPOAgent
from utils import env_fn, Batch, ExpTuple, PPOBuffer, get_logger, get_lr_scheduler, add_git_info


#####################
# Utility Functions #
#####################
def eval_policy(agent, env, eval_episodes: int = 10) -> Tuple[float]:
    """For-loop sequential evaluation."""
    t1 = time.time()
    avg_reward = 0.
    eval_step = 0
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action, _ = agent.sample_actions(obs[None, ...], eval_mode=True)
            next_obs, reward, done, _ = env.step(action.squeeze().clip(-0.99999, 0.99999))
            avg_reward += reward
            eval_step += 1
            obs = next_obs
    avg_reward /= eval_episodes
    return avg_reward, eval_step, time.time() - t1


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_prefix = f"ppo_N{config.actor_num}_L{config.config.rollout_len}"
    exp_name = f"s{config.seed}_{timestamp}"
    exp_info = f"# Running experiment for: {exp_prefix}_{exp_name}_{config.env_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))

    os.makedirs(f"logs/{config.env_name}", exist_ok=True)
    logger = get_logger(f"logs/{config.env_name}/{exp_name}.log")
    add_git_info(config)
    logger.info(f"Exp configurations:\n{config}")

    # set random seed (deterministic on cpu)
    np.random.seed(config.seed)

    # determine training steps
    trajectory_len = config.actor_num * config.rollout_len
    batch_size = config.batch_size
    batch_num = trajectory_len // batch_size
    iterations_per_step = trajectory_len // batch_size

    loop_steps = config.total_steps // trajectory_len
    assert (config.total_steps % trajectory_len == 0) and ((config.total_steps // trajectory_len) % config.log_num == 0)
    log_steps = loop_steps // config.log_num

    # Initialize envpool envs
    train_envs = gym.vector.SyncVectorEnv([env_fn(config.env_name, seed=i) for i in range(config.actor_num)])
    eval_env = env_fn(config.env_name, seed=config.seed*100)()
    act_dim = eval_env.action_space.shape[0]
    obs_dim = eval_env.observation_space.shape[0]
    buffer = PPOBuffer(obs_dim,
                       act_dim,
                       config.rollout_len,
                       config.actor_num,
                       config.gamma,
                       config.lmbda)

    print(f"Total_steps={config.total_steps/1e6:.0f}M steps, actor_num={config.actor_num}, "
          f"trajectory_len={trajectory_len}, act_dim={act_dim}, obs_dim={obs_dim}.")

    # initialize lr scheduler
    lr = get_lr_scheduler(config, loop_steps, iterations_per_step)

    # initialize PPOAgent
    agent = PPOAgent(config, obs_dim, act_dim, lr)
    logs = [{"step": 0, "reward": eval_policy(agent, eval_env)[0]}]

    # reset environments
    observations = train_envs.reset()

    # start training
    for step in trange(loop_steps, desc="[Loop steps]"):
        step_time = time.time()
        all_experiences = []

        for _ in range(config.rollout_len + 1):
            actions, log_probs = agent.sample_actions(observations)
            values = agent.get_values(observations)
            next_observations, rewards, dones, _ = train_envs.step(actions.clip(-0.99999, 0.99999))
            experiences = [ExpTuple(observations[i], actions[i], rewards[i],
                                    values[i], log_probs[i], dones[i])
                           for i in range(config.actor_num)]
            all_experiences.append(experiences)
            observations = next_observations

        # add to trajectory buffer
        buffer.add_experiences(all_experiences)
        trajectory_batch = buffer.process_experience()

        # train sampled trajectories for K epochs
        for _ in range(config.num_epochs):
            permutation = np.random.permutation(trajectory_len)
            approx_kl = 0.
            for i in range(batch_num):
                batch_idx = permutation[i * batch_size:(i + 1) * batch_size]
                batch = Batch(
                    observations=trajectory_batch.observations[batch_idx],
                    actions=trajectory_batch.actions[batch_idx],
                    log_probs=trajectory_batch.log_probs[batch_idx],
                    targets=trajectory_batch.targets[batch_idx],
                    advantages=trajectory_batch.advantages[batch_idx])
                log_info = agent.update(batch)
                approx_kl += log_info["approx_kl"].item()

            # early stopping
            approx_kl /= batch_num
            if approx_kl > 1.5 * config.target_kl:
                break

        # evaluate
        if (step + 1) % log_steps == 0:
            step_num = (step + 1) * trajectory_len // 1_000
            step_fps = trajectory_len / (time.time() - step_time)
            eval_reward, eval_step, eval_time = eval_policy(agent, eval_env)
            eval_fps = eval_step / eval_time
            elapsed_time = (time.time() - start_time) / 60
            log_info.update({
                "step": step_num,
                "reward": eval_reward,
                "time": elapsed_time,
                "step_fps": step_fps,
                "eval_fps": eval_fps,
            })
            logs.append(log_info)
            logger.info(
                f"\n#Step {step_num}K: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, eval_fps={eval_fps:.2f}\n"
                f"\ttotal_time={elapsed_time:.2f}min, step_fps={step_fps:.2f}\n"
                f"\tvalue_loss={log_info['value_loss']:.3f}, ppo_loss={log_info['ppo_loss']:.3f}, "
                f"entropy_loss={log_info['entropy_loss']:.3f}, total_loss={log_info['total_loss']:.3f}\n"
                f"\tavg_target={log_info['avg_target']:.3f}, max_target={log_info['max_target']:.3f}, min_target={log_info['min_target']:.3f}\n"
                f"\tavg_value={log_info['avg_value']:.3f}, max_value={log_info['max_value']:.3f}, min_value={log_info['min_value']:.3f}\n"
                f"\tavg_logp={log_info['avg_logp']:.3f}, max_logp={log_info['max_logp']:.3f}, min_logp={log_info['min_logp']:.3f}\n"
                f"\tavg_old_logp={log_info['avg_old_logp']:.3f}, max_old_logp={log_info['max_old_logp']:.3f}, min_old_logp={log_info['min_old_logp']:.3f}\n"
                f"\tavg_ratio={log_info['avg_ratio']:.3f}, max_ratio={log_info['max_ratio']:.3f}, min_ratio={log_info['min_ratio']:.3f}\n"
                f"\tapprox_kl={approx_kl:.3f}, clipped_frac={log_info['clipped_frac']:.3f}\n"
            )

        # Save checkpoints
        # if (step + 1) % ckpt_steps == 0:
        #     agent.save(f"{ckpt_dir}", (step + 1) // ckpt_steps)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{config.log_dir}/{config.env_name}/{exp_name}.csv")
