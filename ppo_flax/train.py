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
from utils import ExpTuple, get_logger


#####################
# Utility Functions #
#####################
def process_experience(experience: List[List[ExpTuple]],
                       actor_steps: int,
                       num_agents: int,
                       gamma: float,
                       lmbda: float,
                       obs_shape: Tuple[int]=(84, 84, 4)):
    """Process experiences for Atari agents: continuous states & discrete actions."""
    states    = np.zeros((actor_steps, num_agents, *obs_shape), dtype=np.float32)
    actions   = np.zeros((actor_steps, num_agents), dtype=np.int32)
    rewards   = np.zeros((actor_steps, num_agents), dtype=np.float32)
    values    = np.zeros((actor_steps+1, num_agents), dtype=np.float32)
    log_probs = np.zeros((actor_steps, num_agents), dtype=np.float32)
    dones     = np.zeros((actor_steps, num_agents), dtype=np.float32)
    assert len(experience) == actor_steps + 1
    for t in range(len(experience) - 1):
        for agent_id, exp_agent in enumerate(experience[t]):
            states[t, agent_id, ...] = exp_agent.state
            actions[t, agent_id] = exp_agent.action
            rewards[t, agent_id] = exp_agent.reward
            values[t, agent_id] = exp_agent.value
            log_probs[t, agent_id] = exp_agent.log_prob
            # Dones need to be 0 for terminal states.
            dones[t, agent_id] = float(not exp_agent.done)

    # experience[-1] for next_values
    for a in range(num_agents):
        values[-1, a] = experience[-1][a].value

    # compute GAE advantage
    advantages = gae_advantages(rewards, dones, values, gamma, lmbda)
    returns = advantages + values[:-1, :]

    # concatenate results
    trajectories = (states, actions, log_probs, returns, advantages)
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(map(
        lambda x: np.reshape(x, (trajectory_len,) +x.shape[2:]), trajectories))
    return trajectories


def eval_policy(ppo_state, env, eval_episodes: int = 10) -> Tuple[float]:
    t1 = time.time()
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False  # obs.shape = (84, 84, 4)
        obs = obs[None, ...]          # add batch dimension ==> (1, 84, 84, 4)
        while not done:
            log_probs, _ = policy_action(ppo_state.apply_fn,
                                         ppo_state.params,
                                         obs)

            probs = np.exp(np.array(log_probs, dtype=np.float32))
            probabilities = probs[0] / probs[0].sum()
            action = np.random.choice(probs.shape[1], p=probabilities)

            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            obs = obs[None, ...] if not done else None
    avg_reward /= eval_episodes
    return avg_reward, time.time() - t1


#############
# PPO Model #
#############
class ActorCritic(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=512, name="hidden", dtype=jnp.float32)(x)
        x = nn.relu(x)

        logits = nn.Dense(features=self.num_outputs, name="logits", dtype=jnp.float32)(x)
        policy_log_probabilities = nn.log_softmax(logits)
        value = nn.Dense(features=1, name="value", dtype=jnp.float32)(x)

        return policy_log_probabilities, value


#################
# Remote Worker #
#################
class RemoteSimulator:
    """Remote actor in a separate process."""

    def __init__(self, env_name: str, rank: int):
        """Start the remote process and create Pipe() to communicate with it."""
        parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=self.rcv_action_send_exp,
                               args=(child_conn, env_name, rank))
        self.proc.daemon = True
        self.conn = parent_conn
        self.proc.start()

    def rcv_action_send_exp(self, conn, env_name: str, rank: int = 0):
        """Run remote actor.

        Receive action from the main learner, perform one step of simulation and
        send back collected experience.
        """
        env = env_utils.create_env(env_name, clip_rewards=True, seed=rank+100)
        while True:
            obs = env.reset()
            done = False
            state = obs[None, ...]
            while not done:
                # (1) send observation to the learner
                conn.send(state)

                # (2) receive sampled action from the learner
                action = conn.recv()

                # (3) interact with the environment
                obs, reward, done, _ = env.step(action)
                next_state = obs[None, ...] if not done else None
                experience = (state, action, reward, done)

                # (4) send next observation to the learner
                conn.send(experience)
                if done:
                    break
                state = next_state


#############
# PPO Agent #
#############
@functools.partial(jax.jit, static_argnums=0)
def policy_action(apply_fn: Callable[..., Any],
                  params: flax.core.frozen_dict.FrozenDict,
                  state: np.ndarray):
    """Sample actions.

    Args:
        apply_fn: the forward function of the actor-critic model
        params: the parameters of the actor-critic model
        state: input observations

    Returns:
        out: a tuple (log_probabilities, values)
    """
    out = apply_fn({"params": params}, state)
    return out


def get_experience(state: train_state.TrainState,
                   simulators: List[RemoteSimulator],
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
        log_probs, values = policy_action(state.apply_fn,
                                          state.params,
                                          simulator_states)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        for i, simulator in enumerate(simulators):
            probabilities = probs[i]
            action = np.random.choice(probs.shape[1], p=probabilities)
            simulator.conn.send(action)

        # (3) receive next states, rewards from remote actors
        experiences = []
        for i, simulator in enumerate(simulators):
            simulator_state, action, reward, done = simulator.conn.recv()
            value = values[i, 0]
            log_prob = log_probs[i][action]
            sample = ExpTuple(simulator_state, action, reward, value, log_prob, done)
            experiences.append(sample)      # List of ExpTuple
        all_experiences.append(experiences)  # List of List of ExpTuple
    return all_experiences


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(rewards: np.ndarray,
                   terminal_masks: np.ndarray,
                   values: np.ndarray,
                   discount: float,
                   gae_param: float):
    """Use GAE to compute advantages.

    As defined by Eq. (11-12) in PPO paper. Implementation uses key observation
    that A_{t} = delta_t + gamma * lmbda * A_{t+1}.
    """
    assert rewards.shape[0] + 1 == values.shape[0], ("One more value needed.")
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t+1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff

        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


#############
# Training  #
#############
def loss_fn(params: flax.core.FrozenDict,
            apply_fn: Callable[..., Any],
            minibatch: Tuple,
            clip_param: float,
            vf_coeff: float,
            entropy_coeff: float):
    """Evaluate the PPO loss function."""
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values = policy_action(apply_fn, params, states)
    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)

    entropy = jnp.sum(-probs*log_probs, axis=1).mean()

    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)

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

    # initialize remote actors (only for env interactions)
    simulators = [RemoteSimulator(config.env_name, i) for i in range(config.num_agents)]

    # initialize PPO model
    num_actions = env_utils.get_num_actions(config.env_name)
    model = ActorCritic(num_outputs=num_actions)

    # initialize params
    key = jax.random.PRNGKey(config.seed)
    ppo_params = model.init(key, jnp.ones((1, 84, 84, 4)))["params"]

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
        all_experiences = get_experience(state=ppo_state,
                                         simulators=simulators,
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
        if (step+1) % 50 == 0:
            eval_reward, eval_time = eval_policy(ppo_state, eval_env)
            logger.info(f"\n#Step {step+1}: eval_reward={eval_reward:.2f}, eval_time={eval_time:.2f}s, "
                        f"total_time={(time.time()-start_time)/60:.2f}min\n"
                        f"\tvalue_loss={log_info['value_loss']:.3f}, pg_loss={log_info['ppo_loss']:.3f}, "
                        f"entropy_loss={log_info['entropy']:.3f}, total_loss={log_info['total_loss']:.3f}\n")

