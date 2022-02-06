import numpy as np
import tensorflow.compat.v1 as tf
import pdb
from static_fn import static_fns


class FakeEnv:
    def __init__(self,
                 model,
                 env_name,
                 penalty_coeff=0.,
                 penalty_learned_var=False,
                 penalty_learned_var_random=False,
                 std_thresh=0.0,
                 per_batch_std_percentile=0.0,
                 oracle=False,
                 oracle_env=None,
                 model_rew_zero=False):
        self.model = model
        self.static_fn = static_fns[env_name]
        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random
        self.model_rew_zero = model_rew_zero

        self.std_thresh = std_thresh
        self.per_batch_std_percentile = per_batch_std_percentile
        self.oracle = oracle
        self.oracle_env = oracle_env
        self.action_space = self.oracle_env.action_space
        self.observation_space = self.oracle_env.observation_space
        self.unwrapped = self.oracle_env.unwrapped

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) +
                             np.log(variances).sum(-1) +
                             (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(
            inputs, factored=True)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
        if self.std_thresh > 0.0:
            if not self.oracle:
                mask = np.amax(np.linalg.norm(ensemble_model_stds[:, :, 1:],
                                              axis=-1),
                               axis=0) < self.std_thresh
                # mask = np.amax(np.linalg.norm(ensemble_model_stds, axis=-1), axis=0) < self.std_thresh
            else:
                true_obs = self.oracle_env._env.step_obs_act(obs, act)
                mask = np.amax(np.linalg.norm(
                    ensemble_model_means[:, :, 1:] - true_obs[None], axis=-1),
                               axis=0) < self.std_thresh
        elif self.per_batch_std_percentile > 0.0:
            per_batch_std_thresh = np.percentile(
                np.amax(np.linalg.norm(ensemble_model_stds[:, :, 1:], axis=-1),
                        axis=0), self.per_batch_std_percentile)
            mask = np.amax(np.linalg.norm(ensemble_model_stds[:, :, 1:],
                                          axis=-1),
                           axis=0) < per_batch_std_thresh
        else:
            mask = None

        if mask is not None:
            ensemble_model_means = ensemble_model_means[:, mask, :]
            ensemble_model_stds = ensemble_model_stds[:, mask, :]
            ensemble_model_vars = ensemble_model_vars[:, mask, :]
            obs = obs[mask]
            act = act[mask]

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            if self.model.reward_classification:
                ensemble_samples = ensemble_model_means.copy()
                ensemble_samples[..., 1:] += np.random.normal(
                    size=ensemble_model_means[
                        ..., 1:].shape) * ensemble_model_stds[..., 1:]
            else:
                ensemble_samples = ensemble_model_means + np.random.normal(
                    size=ensemble_model_means.shape) * ensemble_model_stds

        if not deterministic:
            #### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
            model_inds = self.model.random_inds(batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            model_means = ensemble_model_means[model_inds, batch_inds]
            model_stds = ensemble_model_stds[model_inds, batch_inds]
            ####
        else:
            samples = np.mean(ensemble_samples, axis=0)
            model_means = np.mean(ensemble_model_means, axis=0)
            model_stds = np.mean(ensemble_model_stds, axis=0)

        if self.model.reward_classification:
            log_prob, dev = self._get_logprob(samples[..., 1:],
                                              ensemble_model_means[..., 1:],
                                              ensemble_model_vars[..., 1:])
        else:
            log_prob, dev = self._get_logprob(samples, ensemble_model_means,
                                              ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        if self.model_rew_zero:
            rewards = np.zeros_like(rewards)
        terminals = self.static_fn.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate(
            (model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate(
            (model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]),
            axis=-1)
        if self.model_rew_zero:
            return_means[:, 0] = 0.
            return_stds[:, 0] = 0.

        if self.penalty_coeff != 0:
            if self.oracle:
                true_obs = self.oracle_env._env.step_obs_act(obs, act)
                penalty = np.amax(np.linalg.norm(
                    ensemble_model_means[:, :, 1:] - true_obs, axis=-1),
                                  axis=0)
            elif not self.penalty_learned_var:
                ensemble_means_obs = ensemble_model_means[:, :, 1:]
                mean_obs_means = np.mean(
                    ensemble_means_obs,
                    axis=0)  # average predictions over models
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0, :obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
                penalty = np.max(dists, axis=0)  # max distances over models
            else:
                # penalty = np.mean(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
                if self.penalty_learned_var_random:
                    penalty = np.linalg.norm(model_stds, axis=1)
                else:
                    # use max variance
                    penalty = np.amax(np.linalg.norm(ensemble_model_stds,
                                                     axis=2),
                                      axis=0)
                    # use mean variance
                    # penalty = np.mean(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)

            penalty = np.expand_dims(penalty, 1)
            assert penalty.shape == rewards.shape
            unpenalized_rewards = rewards
            penalized_rewards = rewards - self.penalty_coeff * penalty

            # mean_penalty, mean_reward = np.mean(penalty), np.mean(rewards)
            # print('Average unweighted penalty:', mean_penalty)
            # print('Average weighted penalty:', self.penalty_coeff * mean_penalty)
            # print('Average reward:', mean_reward)
        else:
            penalty = None
            unpenalized_rewards = rewards
            penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            # rewards = rewards[0]
            unpenalized_rewards = unpenalized_rewards[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        info = {
            'mean': return_means,
            'std': return_stds,
            'log_prob': log_prob,
            'dev': dev,
            'unpenalized_rewards': unpenalized_rewards,
            'penalty': penalty,
            'penalized_rewards': penalized_rewards,
            'mask': mask
        }
        if self.model_rew_zero:
            assert np.all(penalized_rewards == 0.)
        return next_obs, penalized_rewards, terminals, info

    def close(self):
        pass

    def reset(self):
        if self.oracle_env:
            return self.oracle_env.reset()
        else:
            # TODO: use initial states in the dataset?
            return None

    def convert_to_active_observation(self, observation):
        if self.oracle_env:
            return self.oracle_env.convert_to_active_observation(observation)
        else:
            return None

    def get_path_infos(self, paths, *args, **kwargs):
        if self.oracle_env:
            return self.oracle_env.get_path_infos(paths, *args, **kwargs)
        else:
            return None
