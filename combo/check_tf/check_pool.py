from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.replay_pools.simple_replay_pool import SimpleReplayPool
import numpy as np
import off_policy.loader as loader
from load_tf_model import load_model

variant = {
    "environment_params": {
        "training": {
        "domain": "hopper",
        "task": "medium-v0",
        "universe": "gym",
        "kwargs": {}
        },
        "evaluation": {
        "domain": "hopper",
        "task": "medium-v0",
        "universe": "gym",
        "kwargs": {}
        }
    },
    "policy_params": {
        "type": "GaussianPolicy",
        "kwargs": {
        "hidden_layer_sizes": [
            256,
            256,
            256
        ],
        "squash": True
        }
    },
    "Q_params": {
        "type": "double_feedforward_Q_function",
        "kwargs": {
        "hidden_layer_sizes": [
            256,
            256,
            256
        ]
        }
    },
    "algorithm_params": {
        "type": "COMBO",
        "universe": "gym",
        "log_dir": "./ray_combo/",
        "kwargs": {
        "epoch_length": 1000,
        "train_every_n_steps": 1,
        "n_train_repeat": 1,
        "eval_render_mode": None,
        "eval_n_episodes": 10,
        "eval_deterministic": True,
        "discount": 0.99,
        "tau": 0.005,
        "reward_scale": 1.0,
        "model_train_freq": 1000,
        "model_retain_epochs": 10,
        "rollout_batch_size": 10000.0,
        "deterministic": False,
        "num_networks": 7,
        "num_elites": 5,
        "real_ratio": 0.5,
        "target_entropy": -3,
        "max_model_t": None,
        "rollout_length": 5,
        "penalty_coeff": 0.0,
        "lr": 3e-05,
        "q_lr": 0.0003,
        "with_min_q": True,
        "min_q_for_real": False,
        "min_q_for_real_and_fake": False,
        "min_q_for_fake_only": False,
        "min_q_for_fake_states": True,
        "backup_with_uniform": False,
        "backup_for_fake_only": False,
        "temp": 1.0,
        "min_q_version": 3,
        "min_q_weight": 3.0,
        "data_subtract": True,
        "num_random": 10,
        "max_q_backup": False,
        "deterministic_backup": True,
        "with_lagrange": False,
        "lagrange_thresh": 5.0,
        "rollout_random": True,
        "multi_step_prediction": False,
        "num_plan_steps": 5,
        "restore": False,
        "cross_validate": False,
        "cross_validate_model_eval": False,
        "cross_validate_eval_n_episodes": 10,
        "cross_validate_n_steps": 5,
        "use_fqe": False,
        "fqe_num_qs": 3,
        "fqe_minq": False,
        "sn": False,
        "separate_mean_var": False,
        "penalty_learned_var": False,
        "pool_load_path": "d4rl/hopper-medium-v0",
        "pool_load_max_size": 2000000,
        "model_name": "hopper-medium_sn_smv_1_0",
        "n_epochs": 3000,
        "n_initial_exploration_steps": 5000,
        "reparameterize": True,
        "target_update_interval": 1,
        "store_extra_policy_info": False,
        "action_prior": "uniform"
        },
        "domain": "hopper",
        "task": "medium-v0",
        "exp_name": "hopper_medium_nolip_len5_minq_3.0_lagrange_0.0_deterministic_backup_temp1.0_lr3e-5_3layers_normalize_for_fake_states_rollout_random_real0.5_iter2000"
    },
    "replay_pool_params": {
        "type": "SimpleReplayPool",
        "kwargs": {
        "max_size": 1000000,
        "obs_filter": False,
        "modify_rew": False
        }
    },
    "sampler_params": {
        "type": "SimpleSampler",
        "kwargs": {
        "max_path_length": 1000,
        "min_pool_size": 1000,
        "batch_size": 256
        }
    },
    "run_params": {
        "seed": 1071,
        "checkpoint_at_end": True,
        "checkpoint_frequency": 300,
        "checkpoint_replay_pool": False
    },
    "restore": None
}
environment_params = variant['environment_params']
training_environment = get_environment_from_params(environment_params['training'])
evaluation_environment = get_environment_from_params(environment_params['evaluation']) if 'evaluation' in environment_params else training_environment
Qs = get_Q_function_from_variant(variant, training_environment)


# Replay buffer ==> SimpleReplayPool
replay_pool = get_replay_pool_from_variant(variant, training_environment)
loader.restore_pool(replay_pool, "d4rl/hopper-medium-v0", 2000000, save_path='./ray_combo/', env=training_environment)

policy = get_policy_from_variant(variant, training_environment, Qs)

# Sampler
sampler = get_sampler_from_variant(variant)

sampler.initialize(training_environment, policy, replay_pool)

# Create model pool
obs_space = replay_pool._observation_space
act_space = replay_pool._action_space
_model_pool = SimpleReplayPool(obs_space, act_space, int(5e5))

fake_env = load_model()

def rollout_model():
    batch = replay_pool.random_batch(10000)
    steps_added = []
    obs = batch['observations']
    for i in range(5):
        act = np.random.uniform(low=-1.0, high=1.0, size=(len(obs), 3))
        next_obs, rew, term, info = fake_env.step(obs, act, False)
        steps_added.append(len(obs))
        samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term}
        _model_pool.add_samples(samples)
        nonterm_mask = ~term.squeeze(-1)
        if nonterm_mask.sum() == 0:
            print('[ Model Rollout ] Breaking early {} : {} | {} / {}'.format(j, i, nonterm_mask.sum(), nonterm_mask.shape))
            break
        obs = next_obs[nonterm_mask]

    mean_rollout_length = sum(steps_added) / 10000
    rollout_stats = {'mean_rollout_length': mean_rollout_length}
    print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
        sum(steps_added), _model_pool.size, _model_pool._max_size, mean_rollout_length, 1))

