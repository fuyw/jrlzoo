from typing import List, Callable, Any, Tuple
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import collections
import functools
import ml_collections
import multiprocessing as mp

import numpy as np

import jax
import jax.numpy as jnp

import optax
import time

import flax
from flax import linen as nn
from flax.training import train_state

import env_utils
from tqdm import trange
from models import ActorCritic, RemoteSimulator
from utils import ExpTuple, get_logger, process_experience


#####################
# Utility Functions #
#####################
@functools.partial(jax.jit, static_argnums=0)
def sample_action(apply_fn: Callable[..., Any],
                  params: flax.core.frozen_dict.FrozenDict,
                  rng: Any,
                  observations: np.ndarray):
    action_distributions, values = apply_fn({"params": params}, observations)
    actions, log_probs = action_distributions.sample_and_log_prob(seed=rng)
    return actions, log_probs, values


def eval_policy(ppo_state, env, rng, eval_episodes: int = 10) -> Tuple[float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False  # obs.shape = (84, 84, 4)
        while not done:
            rng, key = jax.random.split(rng, 2)
            sampled_action, _, _ = sample_action(ppo_state.apply_fn,
                                                 ppo_state.params,
                                                 key,
                                                 obs[None, ...])
            sampled_action = np.asarray(sampled_action)
            next_obs, reward, done, _ = env.step(sampled_action)
            avg_reward += reward
            obs = next_obs
    avg_reward /= eval_episodes
    return rng, avg_reward, time.time() - t1


def get_experience(state: train_state.TrainState,
                   simulators: List[RemoteSimulator],
                   rng: Any,
                   steps_per_actor: int):
    #TODO: inference in the subprocess
    #TODO: use sequential actors
    #TODO: use vec_env
    """Collect experience using remote actors.
    (1) receive states from remote actors.
    (2) sample action locally, and send sampled actions to remote actors.
    (3) receive next states, rewards from remote actors.

    Runs `steps_per_actor` time steps of the game for each of the `simulators`.
    """
    all_experiences = []

    # Range up to steps_per_actor + 1 to get one more value needed for GAE.
    for _ in range(steps_per_actor + 1):
        # (1) receive remote actor states
        simulator_states = []
        for simulator in simulators:
            simulator_state = simulator.conn.recv()
            simulator_states.append(simulator_state)
        simulator_states = np.concatenate(simulator_states, axis=0)

        # (2) sample actions locally, and send sampled actions to remote actors
        rng, key = jax.random.split(rng)
        actions, log_probs, values = sample_action(state.apply_fn,
                                                   state.params,
                                                   key,
                                                   simulator_states)
        # actions, log_probs, values = jax.device_get((actions, log_probs, values))
        actions = np.asarray(actions)
        log_probs = np.asarray(log_probs)
        values = np.asarray(values)
        for i, simulator in enumerate(simulators):
            simulator.conn.send(actions[i])

        # (3) receive next states, rewards from remote actors
        experiences = []
        for i, simulator in enumerate(simulators):
            simulator_state, action, reward, done = simulator.conn.recv()
            value = values[i]
            log_prob = log_probs[i]
            sample = ExpTuple(simulator_state, action, reward, value, log_prob, done)
            experiences.append(sample)      # List of ExpTuple
        all_experiences.append(experiences)  # List of List of ExpTuple
    return rng, all_experiences


#############
# PPO Agent #
#############
def loss_fn(params: flax.core.FrozenDict,
            apply_fn: Callable[..., Any],
            minibatch: Tuple,
            clip_param: float,
            vf_coeff: float,
            entropy_coeff: float):
    """Evaluate the PPO loss function."""
    states, actions, old_log_probs, returns, advantages = minibatch
    action_distributions, values = apply_fn({"params":params}, states)
    #log_probs, values = policy_action(apply_fn, params, states)
    #values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    #probs = jnp.exp(log_probs)
    #entropy = jnp.sum(-probs*log_probs, axis=1).mean()
    #log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    log_probs = action_distributions.log_prob(actions)
    entropy = action_distributions.entropy().mean()
    value_loss = jnp.mean(jnp.square(returns - values), axis=0)
    ratios = jnp.exp(log_probs - old_log_probs)

    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1.-clip_param, ratios, 1.+clip_param)
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    total_loss = ppo_loss + vf_coeff*value_loss - entropy_coeff*entropy
    log_info = {"ppo_loss": ppo_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": total_loss}

    return total_loss, log_info


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state: train_state.TrainState,
               trajectories: Tuple,
               batch_size: int,
               clip_param: float,
               vf_coeff: float,
               entropy_coeff: float):
    iterations = trajectories[0].shape[0] // batch_size
    trajectories = jax.tree_map(lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories)
    loss = 0.
    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (l, log_info), grads = grad_fn(state.params, state.apply_fn, batch, clip_param, vf_coeff, entropy_coeff)
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss, log_info


def get_lr_scheduler(config, loop_steps, iterations_per_step):
    # set lr scheduler
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(init_value=config.lr,
                                   end_value=0.,
                                   transition_steps=loop_steps*config.num_epochs*iterations_per_step)
    else:
        lr = config.lr
    return lr


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_name = f"ppo_s{config.seed}_{timestamp}"
    exp_info = f"# Running experiment for: {exp_name}_{config.env_name} #"
    logger = get_logger(f"logs/{config.env_name}/{exp_name}.log")
    logger.info(f"Exp configurations:\n{config}")

    # initialize eval env
    eval_env = env_utils.create_env(config.env_name, clip_rewards=False)
    act_dim = eval_env.preproc.action_space.n

    # initialize remote actors (only for env interactions)
    simulators = [RemoteSimulator(config.env_name, i) for i in range(config.num_agents)]

    # initialize PPO model
    model = ActorCritic(act_dim=act_dim)

    # initialize params
    rng = jax.random.PRNGKey(config.seed)
    rng, rollout_rng, eval_rng = jax.random.split(rng, 3)
    ppo_params = model.init(rng, jnp.ones((1, 84, 84, 4)))["params"]

    # determine training steps
    loop_steps = config.total_frames // (config.num_agents * config.actor_steps)
    iterations_per_step = (config.num_agents * config.actor_steps // config.batch_size)

    # create train state
    lr = get_lr_scheduler(config, loop_steps, iterations_per_step)
    ppo_state = train_state.TrainState.create(apply_fn=model.apply,
                                              params=ppo_params,
                                              tx=optax.adam(lr))

    # start training
    for step in trange(loop_steps, desc="[Loop steps]"):
        rollout_rng, all_experiences = get_experience(state=ppo_state,
                                         simulators=simulators,
                                         rng=rollout_rng,
                                         steps_per_actor=config.actor_steps)
        trajectories = process_experience(experience=all_experiences,
                                          actor_steps=config.actor_steps,
                                          num_agents=config.num_agents,
                                          gamma=config.gamma,
                                          lmbda=config.lmbda)
        alpha = 1. - step / loop_steps if config.decaying_lr_and_clip_param else 1.
        clip_param = config.clip_param * alpha

        for _ in range(config.num_epochs):
            permutation = np.random.permutation(config.num_agents * config.actor_steps)
            trajectories = tuple(x[permutation] for x in trajectories)
            ppo_state, _, log_info = train_step(ppo_state,
                                                trajectories,
                                                config.batch_size,
                                                clip_param=clip_param,
                                                vf_coeff=config.vf_coeff,
                                                entropy_coeff=config.entropy_coeff)

        # evaluate
        if (step+1) % 100 == 0:
            eval_rng, eval_reward, eval_time = eval_policy(ppo_state, eval_env, eval_rng)
            logger.info(f"\n#Step {step+1}: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                        f"total_time={(time.time()-start_time)/60:.2f}min\n"
                        f"\tvalue_loss={log_info['value_loss']:.3f}, pg_loss={log_info['ppo_loss']:.3f}, "
                        f"entropy_loss={log_info['entropy']:.3f}, total_loss={log_info['total_loss']:.3f}\n")
            print(f"\n#Step {step+1}: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                  f"total_time={(time.time()-start_time)/60:.2f}min\n"
                  f"\tvalue_loss={log_info['value_loss']:.3f}, pg_loss={log_info['ppo_loss']:.3f}, "
                  f"entropy_loss={log_info['entropy']:.3f}, total_loss={log_info['total_loss']:.3f}\n")
