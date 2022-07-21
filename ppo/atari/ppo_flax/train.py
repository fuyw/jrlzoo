from typing import Tuple
import os
import ml_collections
import numpy as np
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


def get_experience(agent, steps_per_actor: int):
    # TODO: inference in the subprocess
    # TODO: use sequential actors
    # TODO: use vec_env
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
        observations = np.concatenate(observations, axis=0)

        # (2) sample actions locally, and send sampled actions to remote actors
        log_probs, values = agent._sample_action(
            agent.learner_state.params, observations)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        for i, actor in enumerate(agent.actors):
            action = np.random.choice(probs.shape[1], p=probs[i])
            actor.conn.send(action)

        # (3) receive next states, rewards from remote actors
        experiences = []
        for i, actor in enumerate(agent.actors):
            observation, action, reward, done = actor.conn.recv()
            value = values[i]
            log_prob = log_probs[i][action]
            sample = ExpTuple(observation, action, reward, value, log_prob, done)
            experiences.append(sample)      # List of ExpTuple
        all_experiences.append(experiences)  # List of List of ExpTuple
    return all_experiences


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"ppo_s{config.seed}_{timestamp}"
    print(f"# Running experiment for: {exp_name}_{config.env_name} #")
    logger = get_logger(f"logs/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # initialize eval env
    eval_env = env_utils.create_env(config.env_name, clip_rewards=False)
    act_dim = eval_env.preproc.action_space.n

    # determine training steps
    loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
    iterations_per_step = (config.num_agents * config.actor_steps // config.batch_size)

    # initialize lr scheduler
    lr = get_lr_scheduler(config, loop_steps, iterations_per_step)

    # initialize PPOAgent
    agent = PPOAgent(config, act_dim, lr)
 
    # start training
    for step in trange(loop_steps, desc="[Loop steps]"):
        alpha = 1. - step / loop_steps if config.decaying_lr_and_clip_param else 1.
        clip_param = config.clip_param * alpha

        all_experiences = get_experience(agent, steps_per_actor=config.actor_steps)
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
        if (step+1) % 100 == 0:
            eval_reward, eval_time = eval_policy(agent, eval_env) 
            logger.info(f"\n#Step {step+1}: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                        f"total_time={(time.time()-start_time)/60:.2f}min\n"
                        f"\tvalue_loss={log_info['value_loss']:.3f}, pg_loss={log_info['ppo_loss']:.3f}, "
                        f"entropy_loss={log_info['entropy']:.3f}, total_loss={log_info['total_loss']:.3f}\n")
            print(f"\n#Step {step+1}: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                  f"total_time={(time.time()-start_time)/60:.2f}min\n"
                  f"\tvalue_loss={log_info['value_loss']:.3f}, pg_loss={log_info['ppo_loss']:.3f}, "
                  f"entropy_loss={log_info['entropy']:.3f}, total_loss={log_info['total_loss']:.3f}\n")
