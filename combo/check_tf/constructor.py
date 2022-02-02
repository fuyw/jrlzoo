import numpy as np
import tensorflow.compat.v1 as tf
import pdb

from fc import FC
from bnn import BNN


def construct_model(obs_dim=11,
                    act_dim=3,
                    rew_dim=1,
                    hidden_dim=200,
                    num_layers=3,
                    model_lr=1e-3,
                    num_networks=7,
                    num_elites=5,
                    session=None,
                    model_type='mlp',
                    sn=False,
                    gradient_penalty=0.0,
                    gradient_penalty_scale=10.0,
                    separate_mean_var=False,
                    no_sn_last=False,
                    name=None,
                    load_dir=None,
                    deterministic=False,
                    multi_step_prediction=False,
                    num_plan_steps=1,
                    reward_classification=False):
    if name is None:
        name = 'BNN'
    print(
        '[ BNN ] Name {} | Observation dim {} | Action dim: {} | Hidden dim: {}'
        .format(name, obs_dim, act_dim, hidden_dim))
    params = {
        'name': name,
        'num_networks': num_networks,
        'num_elites': num_elites,
        'gradient_penalty': gradient_penalty,
        'gradient_penalty_scale': gradient_penalty_scale,
        'sess': session,
        'separate_mean_var': separate_mean_var,
        'deterministic': deterministic,
        'multi_step_prediction': multi_step_prediction,
        'num_plan_steps': num_plan_steps,
        'obs_dim': obs_dim,
        'reward_classification': reward_classification,
        'load_model': True,
    }

    if load_dir is not None:
        print('Specified load dir', load_dir)
        params['model_dir'] = load_dir

    model = BNN(params)
    if not model.model_loaded:
        if model_type == 'identity':
            return
        elif model_type == 'linear':
            print('[ BNN ] Training linear model')
            model.add(
                FC(obs_dim + rew_dim,
                   input_dim=obs_dim + act_dim,
                   weight_decay=0.000025))
        elif model_type == 'mlp':
            print(
                '[ BNN ] Training non-linear model | Obs: {} | Act: {} | Rew: {} | Hidden: {}'
                .format(obs_dim, act_dim, rew_dim, hidden_dim))
            model.add(
                FC(hidden_dim,
                   input_dim=obs_dim + act_dim,
                   activation="swish",
                   weight_decay=0.000025,
                   sn=sn))
            model.add(
                FC(hidden_dim, activation="swish", weight_decay=0.00005,
                   sn=sn))
            assert num_layers > 2
            for i in range(num_layers - 2):
                model.add(
                    FC(hidden_dim,
                       activation="swish",
                       weight_decay=0.000075,
                       sn=sn))
            if no_sn_last:
                model.add(FC(obs_dim + rew_dim, weight_decay=0.0001, sn=False))
            else:
                model.add(FC(obs_dim + rew_dim, weight_decay=0.0001, sn=sn))
            if separate_mean_var:
                model.add(FC(obs_dim + rew_dim,
                             input_dim=hidden_dim,
                             weight_decay=0.0001,
                             sn=False),
                          var_layer=True)
        elif model_type == 'recurrent':
            pass

    if load_dir is not None:
        # janky hack because the model structure saving/loading was not working
        model.model_loaded = True

    # model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": model_lr})
    print('[ BNN ] Model: {}'.format(model))
    return model


def format_samples_for_training(samples,
                                multi_step_prediction=False,
                                num_plan_steps=1):
    obs = samples['observations']
    act = samples['actions']
    next_obs = samples['next_observations']
    rew = samples['rewards']
    if not multi_step_prediction:
        delta_obs = next_obs - obs
        inputs = np.concatenate((obs, act), axis=-1)
        outputs = np.concatenate((rew, delta_obs), axis=-1)
    else:
        num_paths = 0
        temp = 0
        min_path_len = np.inf
        path_end_idx = []
        for i in range(samples['terminals'].shape[0]):
            if samples['terminals'][i] or i - temp + 1 == 1000:
                min_path_len = min(min_path_len, i - temp + 1)
                num_paths += 1
                temp = i + 1
                path_end_idx.append(i)
        print('minimum path length is %d' % min_path_len)
        assert min_path_len >= num_plan_steps
        inputs, outputs = [], []
        for i in range(num_paths):
            inputs_sub, outputs_sub = [], []
            if i == 0:
                path_start_idx = 0
            else:
                path_start_idx = path_end_idx[i - 1] + 1
            for j in range(path_start_idx,
                           path_end_idx[i] - num_plan_steps + 2):
                inputs_sub.append(
                    np.concatenate(
                        (obs[j:j + num_plan_steps], act[j:j + num_plan_steps]),
                        axis=-1))
                outputs_sub.append(
                    np.concatenate((rew[j:j + num_plan_steps],
                                    next_obs[j:j + num_plan_steps] -
                                    obs[j:j + num_plan_steps]),
                                   axis=-1))
                try:
                    assert not np.any(
                        samples['terminals'][j:j + num_plan_steps - 1])
                except:
                    import pdb
                    pdb.set_trace()
            inputs.extend(inputs_sub)
            outputs.extend(outputs_sub)
        inputs = np.array(inputs)
        outputs = np.array(outputs)
    return inputs, outputs


def reset_model(model):
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope=model.name)
    model.sess.run(tf.initialize_vars(model_vars))


if __name__ == '__main__':
    model = construct_model()
