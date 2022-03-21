import d4rl
import gym
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dynamics_model import DynamicsModel


def compute_obs_loss(true_obs, pred_obs):
    delta_obs = jnp.abs(true_obs - pred_obs)
    obs_loss = delta_obs.sum(1).mean(0)
    return obs_loss


def compute_rew_loss(true_reward, pred_reward):
    delta_reward = jnp.abs(true_reward.squeeze() - pred_reward.squeeze())
    rew_loss = delta_reward.mean(0)
    return rew_loss


def load_buffer(env_name):
    data = np.load(f"saved_buffers/{env_name.split('-')[0]}-v2/L100K.npz")
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    return observations, actions, next_observations, rewards


def step(model, observation, action):
    x = jnp.concatenate([observation, action], axis=-1).reshape(1, -1)
    model_mu, _ = model.model.apply({"params": model.model_state.params}, x)
    observation_mu, reward_mu = jnp.split(model_mu, [model.obs_dim], -1)  # (7, 11),  (7, 1)
    next_observation = observation + observation_mu
    next_observation = model.denormalize(next_observation)  # (7, 11)
    return next_observation.mean(0), reward_mu.mean(0)
step = jax.vmap(step, in_axes=(None, 0, 0))


def eval_model(model):
    normalized_obs = model.normalize(observations)
    pred_next_observations, pred_rewards = step(model, normalized_obs, actions)  # (100000, 11),  (100000, 1)
    obs_loss = compute_obs_loss(observations, pred_next_observations)  # (19.158)
    rew_loss = compute_rew_loss(rewards, pred_rewards)
    return obs_loss, rew_loss


res = []
for env_name in [f"{i}-{j}-v2" for i in ["hopper", "halfcheetah", "walker2d"] for j in [
        "medium", "medium-replay", "medium-expert"]]:
    dynamics_model = DynamicsModel(env_name)
    observations, actions, next_observations, rewards = load_buffer(env_name)
    dynamics_model.load_old()
    old_obs_loss, old_rew_loss = eval_model(dynamics_model)
    res.append((env_name, "old", old_obs_loss, old_rew_loss))

    dynamics_model.load_new()
    new_obs_loss, new_rew_loss = eval_model(dynamics_model)
    res.append((env_name, "new", new_obs_loss, new_rew_loss))
res_df = pd.DataFrame(res, columns=['env_name', 'model', 'obs_loss', 'rew_loss'])
