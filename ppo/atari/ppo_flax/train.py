from typing import List, Tuple
import os
import ml_collections
import numpy as np
import pandas as pd
import jax
import time
import envpool
import env_utils
from tqdm import trange
from models import PPOAgent
from utils import Batch, ExpTuple, PPOBuffer, get_logger, get_lr_scheduler

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"


#####################
# Utility Functions #
#####################
def eval_policy(agent: PPOAgent,
                eval_envs: envpool.atari.AtariGymEnvPool, 
                eval_episodes: int = 10) -> Tuple[float]:
    """Evaluate with envpool vectorized environments."""
    t1 = time.time()
    n_envs = len(eval_envs.all_env_ids)
    episode_rewards = []
    episode_lengths = []
    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([
        (eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")

    observations = eval_envs.reset()
    while (episode_counts < episode_count_targets).any():
        # (10, 4, 84, 84) ==> (10, 84, 84, 4)
        actions = agent.sample_actions(np.moveaxis(observations, 1, -1))
        observations, rewards, dones, _ = eval_envs.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
    avg_reward = np.mean(episode_rewards)
    avg_step = np.mean(episode_lengths)
    return avg_reward, avg_step, time.time() - t1


def get_experience(agent, steps_per_actor: int) -> List[List[ExpTuple]]:
    """Collect experience using remote actors.
    (1) receive states from remote actors.
    (2) sample action locally, and send sampled actions to remote actors.
    (3) receive next states, rewards from remote actors.

    Runs `steps_per_actor` time steps of the game for each of the `agent.actors`.
    """
    all_experiences = []

    # Range up to steps_per_actor + 1 to get one more value needed for GAE.
    for _ in range(steps_per_actor + 1):
        # (1) receive remote actor states
        observations = []
        for actor in agent.actors:
            observation = actor.conn.recv()
            observations.append(observation)
        observations = np.concatenate(observations, axis=0)  # (5, 84, 84, 4)

        # (2) sample actions locally, and send sampled actions to remote actors
        log_probs, values = agent._sample_actions(agent.learner_state.params,
                                                  observations)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        actions = [np.random.choice(len(prob), p=prob) for prob in probs]
        for i, actor in enumerate(agent.actors):
            actor.conn.send(actions[i])

        # (3) receive next states, rewards from remote actors
        experiences = []
        for i, actor in enumerate(agent.actors):
            reward, done = actor.conn.recv()
            sample = ExpTuple(observation=observations[i],
                              action=actions[i],
                              reward=reward,
                              value=values[i],
                              log_prob=log_probs[i][actions[i]],
                              done=done)
            experiences.append(sample)  # List of ExpTuple for each actor
        all_experiences.append(experiences)  # List of List of ExpTuple
    return all_experiences


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"ppo_s{config.seed}_a{config.actor_num}_{timestamp}"
    print(f"# Running experiment for: {exp_name}_{config.env_name} #")
    logger = get_logger(f"logs/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # set random seed (deterministic on cpu)
    np.random.seed(config.seed)

    # initialize eval environment
    # eval_env = env_utils.create_env(config.env_name,
    #                                 clip_rewards=False,
    #                                 seed=config.seed)
    eval_env = envpool.make(
        task_id=f"{config.env_name.split('NoFrameskip')[0]}-v5",
        env_type="gym",
        num_envs=10,
        episodic_life=False,
        reward_clip=False,
    )
    act_dim = eval_env.action_space.n

    # determine training steps
    trajectory_len = config.actor_num * config.rollout_len
    batch_size = config.batch_size
    batch_num = trajectory_len // batch_size
    iterations_per_step = trajectory_len // batch_size
    loop_steps = config.total_frames // trajectory_len
    assert config.total_frames % trajectory_len % config.log_num == 0
    log_steps = loop_steps // config.log_num

    # initialize lr scheduler
    lr = get_lr_scheduler(config, loop_steps, iterations_per_step)

    # initialize PPOAgent
    agent = PPOAgent(config, act_dim, lr)
    buffer = PPOBuffer(config.rollout_len, config.actor_num, config.gamma,
                       config.lmbda)
    logs = [{"frame": 0, "reward": eval_policy(agent, eval_env)[0]}]

    # start training
    for step in trange(loop_steps, desc="[Loop steps]"):
        all_experiences = get_experience(agent,
                                         steps_per_actor=config.rollout_len)
        buffer.add_experiences(all_experiences)
        trajectory_batch = buffer.process_experience()

        # train sampled trajectories for K epochs
        for _ in range(config.num_epochs):
            permutation = np.random.permutation(trajectory_len)
            for i in range(batch_num):
                batch_idx = permutation[i * batch_size:(i + 1) * batch_size]
                batch = Batch(observations=trajectory_batch.observations[batch_idx],
                              actions=trajectory_batch.actions[batch_idx],
                              log_probs=trajectory_batch.log_probs[batch_idx], 
                              targets=trajectory_batch.targets[batch_idx],
                              advantages=trajectory_batch.advantages[batch_idx])
                log_info = agent.update(batch)

        # evaluate
        if (step + 1) % log_steps == 0:
            frame_num = (step + 1) * trajectory_len // 1_000
            eval_reward, eval_step, eval_time = eval_policy(agent, eval_env)
            eval_fps = eval_step / eval_time
            elapsed_time = (time.time() - start_time) / 60
            log_info.update({
                "frame": frame_num,
                "reward": eval_reward,
                "time": elapsed_time,
                "eval_fps": eval_fps,
            })
            logs.append(log_info)
            logger.info(
                f"\n#Frame {frame_num}K: eval_reward={eval_reward:.2f}, eval_fps={eval_fps:.2f}, "
                f"eval_time={eval_time:.2f}s, total_time={elapsed_time:.2f}min\n"
                f"\tvalue_loss={log_info['value_loss']:.3f}, ppo_loss={log_info['ppo_loss']:.3f}, "
                f"entropy_loss={log_info['entropy_loss']:.3f}, total_loss={log_info['total_loss']:.3f}\n"
                f"\tavg_target={log_info['avg_target']:.3f}, max_target={log_info['max_target']:.3f}, min_target={log_info['min_target']:.3f}\n"
                f"\tavg_value={log_info['avg_value']:.3f}, max_value={log_info['max_value']:.3f}, min_value={log_info['min_value']:.3f}\n"
                f"\tavg_logp={log_info['avg_logp']:.3f}, max_logp={log_info['max_logp']:.3f}, min_logp={log_info['min_logp']:.3f}\n"
                f"\tavg_ratio={log_info['avg_ratio']:.3f}, max_ratio={log_info['max_ratio']:.3f}, min_ratio={log_info['min_ratio']:.3f}\n"
            )
            print(
                f"\n#Frame {frame_num}K: eval_reward={eval_reward:.2f}, eval_fps={eval_fps:.2f}, "
                f"eval_time={eval_time:.2f}s, total_time={elapsed_time:.2f}min\n"
                f"\tvalue_loss={log_info['value_loss']:.3f}, ppo_loss={log_info['ppo_loss']:.3f}, "
                f"entropy_loss={log_info['entropy_loss']:.3f}, total_loss={log_info['total_loss']:.3f}\n"
                f"\tavg_target={log_info['avg_target']:.3f}, max_target={log_info['max_target']:.3f}, min_target={log_info['min_target']:.3f}\n"
                f"\tavg_value={log_info['avg_value']:.3f}, max_value={log_info['max_value']:.3f}, min_value={log_info['min_value']:.3f}\n"
                f"\tavg_logp={log_info['avg_logp']:.3f}, max_logp={log_info['max_logp']:.3f}, min_logp={log_info['min_logp']:.3f}\n"
                f"\tavg_ratio={log_info['avg_ratio']:.3f}, max_ratio={log_info['max_ratio']:.3f}, min_ratio={log_info['min_ratio']:.3f}\n"
            )

    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"{config.log_dir}/{config.env_name}/{exp_name}.csv")
