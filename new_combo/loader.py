import os
import glob
import pickle
import gzip
import pdb
import numpy as np

def restore_pool(replay_pool, experiment_root, max_size, save_path=None, eval_pool=False, env=None):
    if 'd4rl' in experiment_root and not eval_pool:
        assert env is not None
        restore_pool_d4rl(replay_pool, experiment_root[5:], env)
    else:
        assert os.path.exists(experiment_root)
        if 'metaworld' in experiment_root:
            restore_pool_metaworld(replay_pool, experiment_root, max_size, save_path)
        elif os.path.isdir(experiment_root):
            restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path)
        else:
            try:
                restore_pool_contiguous(replay_pool, experiment_root)
            except:
                restore_pool_bear(replay_pool, experiment_root)
    print('[ combo/off_policy ] Replay pool has size: {}'.format(replay_pool.size))


def restore_pool_d4rl(replay_pool, name, env):
    import gym
    import d4rl
    # if 'kitchen' in name:
    #     data = gym.make(name).unwrapped.get_dataset()
    #     data['next_observations'] = data['observations'][1:]
    # else:
    data = d4rl.qlearning_dataset(env.unwrapped)
    # import pdb; pdb.set_trace()
    if 'antmaze' in name or 'kitchen' in name:
        data['rewards'] = np.expand_dims(data['rewards'], axis=1)
        # import pdb; pdb.set_trace()
        # data['rewards'] = (np.expand_dims(data['rewards'], axis=1) - 0.5) * 4.0
    elif 'pen' in name:
        data['rewards'] = np.expand_dims(data['rewards'], axis=1)*0.02 - 0.5
    elif 'hammer' in name:
        data['rewards'] = np.expand_dims(data['rewards'], axis=1)*0.02 - 0.05
    elif 'door' in name:
        data['rewards'] = np.expand_dims(data['rewards'], axis=1)*0.1
    else:
        data['rewards'] = (np.expand_dims(data['rewards'], axis=1) - 0.5) * 4.0
    # data['rewards'] = (np.expand_dims(data['rewards'], axis=1) - 0.5) * 20.0
    data['terminals'] = np.expand_dims(data['terminals'], axis=1)
    # if 'kitchen' in name:
    #     data['observations'] = data['observations'][:-1]
    #     data['actions'] = data['actions'][:-1]
    #     data['rewards'] = data['rewards'][:-1]
    #     data['terminals'] = data['terminals'][:-1]
    replay_pool.add_samples(data)


def restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path=None):
    print('[ combo/off_policy ] Loading SAC replay pool from: {}'.format(experiment_root))
    experience_paths = [
        checkpoint_dir
        for checkpoint_dir in sorted(glob.iglob(
            os.path.join(experiment_root, 'checkpoint_*')))
    ]

    checkpoint_epochs = [int(path.split('_')[-1]) for path in experience_paths]
    checkpoint_epochs = sorted(checkpoint_epochs)
    if max_size == 250e3:
        checkpoint_epochs = checkpoint_epochs[2:]

    for epoch in checkpoint_epochs:
        fullpath = os.path.join(experiment_root, 'checkpoint_{}'.format(epoch), 'replay_pool.pkl')
        print('[ combo/off_policy ] Loading replay pool data: {}'.format(fullpath))
        replay_pool.load_experience(fullpath)
        if replay_pool.size >= max_size:
            break

    if save_path is not None:
        size = replay_pool.size
        stat_path = os.path.join(save_path, 'pool_stat_{}.pkl'.format(size))
        save_path = os.path.join(save_path, 'pool_{}.pkl'.format(size))
        d = {}
        for key in replay_pool.fields.keys():
            d[key] = replay_pool.fields[key][:size]

        num_paths = 0
        temp = 0
        path_end_idx = []
        for i in range(d['terminals'].shape[0]):
            if d['terminals'][i] or i - temp + 1 == 1000:
                num_paths += 1
                temp = i + 1
                path_end_idx.append(i)
        total_return = d['rewards'].sum()
        avg_return = total_return / num_paths
        buffer_max, buffer_min = -np.inf, np.inf
        path_return = 0.0
        for i in range(d['rewards'].shape[0]):
            path_return += d['rewards'][i]
            if i in path_end_idx:
                if path_return > buffer_max:
                    buffer_max = path_return
                if path_return < buffer_min:
                    buffer_min = path_return
                path_return = 0.0

        print('[ combo/off_policy ] Replay pool average return is {}, buffer_max is {}, buffer_min is {}'.format(avg_return, buffer_max, buffer_min))
        d_stat = dict(avg_return=avg_return, buffer_max=buffer_max, buffer_min=buffer_min)
        pickle.dump(d_stat, open(stat_path, 'wb'))

        print('[ combo/off_policy ] Saving replay pool to: {}'.format(save_path))
        pickle.dump(d, open(save_path, 'wb'))

    
    ####
    # val_size = 1000
    # print('NOT USING LAST {} SAMPLES'.format(val_size))
    # replay_pool._pointer -= val_size
    # replay_pool._size -= val_size
    # print(replay_pool._pointer, replay_pool._size)
    # pdb.set_trace()


def restore_pool_bear(replay_pool, load_path):
    print('[ combo/off_policy ] Loading BEAR replay pool from: {}'.format(load_path))
    data = pickle.load(gzip.open(load_path, 'rb'))
    num_trajectories = data['terminals'].sum() or 1000
    avg_return = data['rewards'].sum() / num_trajectories
    print('[ combo/off_policy ] {} trajectories | avg return: {}'.format(num_trajectories, avg_return))

    for key in ['log_pis', 'data_policy_mean', 'data_policy_logvar']:
        del data[key]

    replay_pool.add_samples(data)


def restore_pool_contiguous(replay_pool, load_path):
    print('[ combo/off_policy ] Loading contiguous replay pool from: {}'.format(load_path))
    import numpy as np
    data = np.load(load_path)

    state_dim = replay_pool.fields['observations'].shape[1]
    action_dim = replay_pool.fields['actions'].shape[1]
    expected_dim = state_dim + action_dim + state_dim + 1 + 1
    actual_dim = data.shape[1]
    assert expected_dim == actual_dim, 'Expected {} dimensions, found {}'.format(expected_dim, actual_dim)

    dims = [state_dim, action_dim, state_dim, 1, 1]
    ends = []
    current_end = 0
    for d in dims:
        current_end += d
        ends.append(current_end)
    states, actions, next_states, rewards, dones = np.split(data, ends, axis=1)[:5]
    replay_pool.add_samples({
        'observations': states,
        'actions': actions,
        'next_observations': next_states,
        'rewards': rewards,
        'terminals': dones.astype(bool)
    })

def restore_pool_metaworld(replay_pool, experiment_root, max_size, save_path=None):
    print('[ combo/off_policy ] Loading metaworld data from: {}'.format(experiment_root))
    import glob
    dataset_keys = ['state', 'rewards', 'actions']
    episode_keys = ['state', 'success', 'action']
    data = {'observations': None, 'actions': None, 'next_observations': None, 'rewards': None, 'terminals': None}
    for idx, filename in enumerate(glob.glob(os.path.join(experiment_root, '*npz'))):
        dataset = {}
        try:
            with open(filename, 'rb') as f:
                episode = np.load(f)
                for k in episode.keys():
                    if k not in episode_keys:
                        continue
                    i = episode_keys.index(k)
                    if dataset_keys[i] in dataset.keys():
                        dataset[dataset_keys[i]] = np.concatenate([dataset[dataset_keys[i]], episode[k]], axis=0).copy()
                    else:
                        dataset[dataset_keys[i]] = episode[k].copy()
                del episode
                if idx % 100 == 0:
                    print('loaded %d files' % idx)
                all_obs = dataset['state']
                all_act = dataset['actions']
                N = min(all_obs.shape[0], 2000000)
                _obs = all_obs[:N-1]
                _actions = all_act[1:N]
                _next_obs = all_obs[1:N]
                _rew = dataset['rewards'][1:N]
                _done = np.zeros_like(_rew)

                if data['observations'] is None:
                    data['observations'] = _obs
                    data['next_observations'] = _next_obs
                    data['actions'] = _actions
                    # Normalize rewards
                    # replay_buffer._rewards = (np.expand_dims(_rew, 1) - 0.5) * 4.0
                    data['rewards'] = np.expand_dims(_rew, 1)
                    data['terminals'] = np.expand_dims(_done, 1) #np.expand_dims(_done, 1)
                else:
                    data['observations'] = np.concatenate((data['observations'], _obs), axis=0)
                    data['next_observations'] = np.concatenate((data['next_observations'], _next_obs), axis=0)
                    data['actions'] = np.concatenate((data['actions'], _actions), axis=0)
                    # Normalize rewards
                    # replay_buffer._rewards = (np.expand_dims(_rew, 1) - 0.5) * 4.0
                    data['rewards'] = np.concatenate((data['rewards'], np.expand_dims(_rew, 1)), axis=0)
                    data['terminals'] = np.concatenate((data['terminals'], np.expand_dims(_done, 1)), axis=0)
        except Exception as e:
            print(f'Could not load episode: {e}')
            continue

    replay_pool.add_samples(data)
    size = replay_pool.size
    d = {}
    for key in replay_pool.fields.keys():
        d[key] = replay_pool.fields[key][:size]
    last_success = d['rewards'].reshape(-1, all_obs.shape[0]-1, 1)[:,-1,:]
    print('[ combo/off_policy ] Replay pool average success is {}, buffer_max is {}, buffer_min is {}'.format(np.mean(last_success), np.amax(last_success), np.amin(last_success)))
    print('[ combo/off_policy ] Loaded metaworld data from: {}'.format(experiment_root))
