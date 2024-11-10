import os
import time

import numpy as np
import torch
from mpi4py import MPI
from tqdm import trange

from her_modules.her import her_sampler
from mpi_utils.mpi_utils import sync_networks, sync_grads
from mpi_utils.normalizer import normalizer
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic


class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.max_action = env_params["max_action"]

        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)

        # the agent in each process have the same parameters
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)

        self.actor_target_network.load_state_dict(
            self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(
            self.critic_network.state_dict())

        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), 
                                            lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), 
                                             lr=self.args.lr_critic)

        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.unwrapped.compute_reward)

        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        self.o_norm = normalizer(size=env_params["obs"], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params["goal"], default_clip_range=self.args.clip_range)

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        log_info = []

        # 50 * 50 * 2 * 100 * 16 / 1e6
        for epoch in trange(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []

                    observation, _ = self.env.reset()
                    obs = observation["observation"]
                    ag = observation["achieved_goal"]
                    g = observation["desired_goal"]

                    for t in range(self.env_params["max_timesteps"]):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)

                        observation_new, _, _, _, info = self.env.step(action)
                        obs_new = observation_new["observation"]
                        ag_new = observation_new["achieved_goal"]

                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    self._update_network()

                self._soft_update_target_network(self.actor_target_network,
                                                 self.actor_network)
                self._soft_update_target_network(self.critic_target_network,
                                                 self.critic_network)

            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"Epoch {epoch+1}: {success_rate:.3f}")
                log_info.append((time.time(), success_rate))
        return log_info

    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)

        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
 
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        action += self.args.noise_eps * self.max_action * np.random.randn(*action.shape)
        action = np.clip(action, -self.max_action, self.max_action)
        random_actions = np.random.uniform(low=-self.max_action,
                                           high=self.max_action,
                                           size=self.env_params["action"])
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        num_transitions = mb_actions.shape[1]
        buffer_temp = {
            "obs": mb_obs,
            "ag": mb_ag,
            "g": mb_g,
            "actions": mb_actions,
            "obs_next": mb_obs_next,
            "ag_next": mb_ag_next,
        }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions["obs"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(obs, g)
        self.o_norm.update(transitions["obs"])
        self.g_norm.update(transitions["g"])
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        transitions = self.buffer.sample(self.args.batch_size)

        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)

        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions["obs_next"])
        g_next_norm = self.g_norm.normalize(transitions["g_next"])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor,
                                                      actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()

            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (
            actions_real / self.max_action).pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation, _ = self.env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]
            for _ in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, _, info = self.env.step(actions)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                per_success_rate.append(info["is_success"])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
