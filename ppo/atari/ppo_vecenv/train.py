from typing import Tuple
import os
import ml_collections
import numpy as np
import pandas as pd
import jax
import time
import env_utils
from tqdm import trange
from models import PPOAgent
from utils import ExpTuple, get_logger, process_experience, get_lr_scheduler

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


#####################
# Utility Functions #
#####################
def eval_policy(agent, env, eval_episodes: int = 10) -> Tuple[float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            log_probs, _ = agent._sample_action(agent.learner_state.params, obs[None, ...])
            log_probs = jax.device_get(log_probs)
            probs = np.exp(log_probs).squeeze()  # (1, act_dim)
            action = np.random.choice(len(probs), p=probs)
            next_obs, reward, done, _ = env.step(action)
            avg_reward += reward
            obs = next_obs
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"ppo_s{config.seed}_a{config.num_agents}_{timestamp}"
    print(f"# Running experiment for: {exp_name}_{config.env_name} #")
    logger = get_logger(f"logs/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # initialize eval env
    vec_env = env_utils.create_vec_env(config.env_name,
                                       num_envs=config.num_agents,
                                       clip_rewards=True,
                                       seeds=range(config.num_agents))
    eval_env = env_utils.create_env(config.env_name, clip_rewards=False)
    act_dim = eval_env.preproc.action_space.n

    # determine training steps
    loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
    iterations_per_step = (config.num_agents * config.actor_steps // config.batch_size)
    log_steps = loop_steps // config.log_num

    # initialize lr scheduler
    lr = get_lr_scheduler(config, loop_steps, iterations_per_step)

    # initialize PPOAgent
    agent = PPOAgent(config, act_dim, lr)
    logs = [{"frame":0, "reward":eval_policy(agent, eval_env)[0]}]

    # reset environments
    observations = vec_env.reset()  # (agent_num, 84, 84, 4)

    # start training
    for step in trange(loop_steps, desc="[Loop steps]"):
        alpha = 1. - step / loop_steps if config.decaying_lr_and_clip_param else 1.
        clip_param = config.clip_param * alpha

        # rollout each actor `actor_steps` to collect trajectories
        all_experiences = []
        for _ in range(config.actor_steps+1):
            log_probs, values = agent._sample_action(agent.learner_state.params, observations)
            log_probs, values = jax.device_get((log_probs, values))
            probs = np.exp(log_probs)  # (agent_num, action_dim)
            actions = np.array([np.random.choice(probs.shape[1], p=prob) for prob in probs])
            next_observations, rewards, dones, _ = vec_env.step(actions)
            experiences = [ExpTuple(observations[i], actions[i], rewards[i],
                                    values[i], log_probs[i][actions[i]], dones[i])
                           for i in range(config.num_agents)]
            all_experiences.append(experiences)
            observations = next_observations

        trajectories = process_experience(experience=all_experiences,
                                          actor_steps=config.actor_steps,
                                          num_agents=config.num_agents,
                                          gamma=config.gamma,
                                          lmbda=config.lmbda)
        iterations = trajectories[0].shape[0] // config.batch_size

        for _ in range(config.num_epochs):
            permutation = np.random.permutation(config.num_agents *
                                                config.actor_steps)
            trajectories = tuple(x[permutation] for x in trajectories)

            batch_trajectories = jax.tree_map(
                lambda x: x.reshape(
                    (iterations, config.batch_size, *x.shape[1:])),
                trajectories)
            for batch in zip(*batch_trajectories):
                log_info = agent.update(batch, clip_param)

        # evaluate
        if (step+1) % log_steps == 0:
            frame_num = (step+1) * config.num_agents * config.actor_steps // 1_000
            eval_reward, eval_time = eval_policy(agent, eval_env) 
            elapsed_time = (time.time()-start_time)/60
            log_info.update({"frame": frame_num, "reward": eval_reward, "time": elapsed_time})
            logs.append(log_info)
            logger.info(f"\n#Frame {frame_num}K: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                        f"total_time={elapsed_time:.2f}min\n"
                        f"\tvalue_loss={log_info['value_loss']:.3f}, ppo_loss={log_info['ppo_loss']:.3f}, "
                        f"entropy_loss={log_info['entropy_loss']:.3f}, total_loss={log_info['total_loss']:.3f}\n")
            print(f"\n#Frame {frame_num}K: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                  f"total_time={elapsed_time:.2f}min\n"
                  f"\tvalue_loss={log_info['value_loss']:.3f}, ppo_loss={log_info['ppo_loss']:.3f}, "
                  f"entropy_loss={log_info['entropy_loss']:.3f}, total_loss={log_info['total_loss']:.3f}\n")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{config.log_dir}/{config.env_name}/{exp_name}.csv")