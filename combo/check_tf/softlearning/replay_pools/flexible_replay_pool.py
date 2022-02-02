import gzip
import pickle

import numpy as np

from .replay_pool import ReplayPool


class FlexibleReplayPool(ReplayPool):
    def __init__(self, max_size, fields_attrs, obs_filter=False, modify_rew=False):
        super(FlexibleReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {}
        self.fields_attrs = {}

        self.add_fields(fields_attrs)

        self.obs_filter = obs_filter
        self.modify_rew = modify_rew

        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0

    @property
    def size(self):
        return self._size

    @property
    def field_names(self):
        return list(self.fields.keys())

    def add_fields(self, fields_attrs):
        self.fields_attrs.update(fields_attrs)

        for field_name, field_attrs in fields_attrs.items():
            field_shape = (self._max_size, *field_attrs['shape'])
            initializer = field_attrs.get('initializer', np.zeros)
            self.fields[field_name] = initializer(
                field_shape, dtype=field_attrs['dtype'])

    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)
        self._samples_since_save += count

    def add_sample(self, sample):
        samples = {
            key: value[None, ...]
            for key, value in sample.items()
        }
        self.add_samples(samples)

    def add_samples(self, samples):
        field_names = list(samples.keys())
        num_samples = samples[field_names[0]].shape[0]

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        if 'reward_angle' not in self.fields_attrs.keys():
            if 'infos' in field_names and 'reward_angle' in samples['infos'][0]:
                assert 'reward_forward' in samples['infos'][0].keys()
                # reward_fields = {
                #     'reward_angle': {
                #         'shape': (1, ),
                #         'dtype': 'float32'
                #     },
                #     'reward_forward': {
                #         'shape': (1, ),
                #         'dtype': 'float32'
                #     },
                # }
                reward_fields = {
                    'reward_angle': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    # 'reward_angle_45': {
                    #     'shape': (1, ),
                    #     'dtype': 'float32'
                    # },
                    'reward_angle_60': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    'reward_angle_90': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    'reward_angle_180': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    'reward_forward': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                }
                self.add_fields(reward_fields)
            elif 'reward_angle' in field_names and 'reward_angle' in samples.keys():
                assert 'reward_forward' in samples.keys()
                # reward_fields = {
                #     'reward_angle': {
                #         'shape': (1, ),
                #         'dtype': 'float32'
                #     },
                #     'reward_forward': {
                #         'shape': (1, ),
                #         'dtype': 'float32'
                #     },
                # }
                reward_fields = {
                    'reward_angle': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    # 'reward_angle_45': {
                    #     'shape': (1, ),
                    #     'dtype': 'float32'
                    # },
                    'reward_angle_60': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    'reward_angle_90': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    'reward_angle_180': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                    'reward_forward': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                }
                self.add_fields(reward_fields)
        if 'reward_jump' not in self.fields_attrs.keys():
            if 'infos' in field_names and 'reward_jump' in samples['infos'][0]:
                reward_fields = {
                    'reward_jump': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                }
                self.add_fields(reward_fields)
            elif 'reward_jump' in field_names and 'reward_jump' in samples.keys():
                reward_fields = {
                    'reward_jump': {
                        'shape': (1, ),
                        'dtype': 'float32'
                    },
                }
                self.add_fields(reward_fields)

        for field_name in self.field_names:
            default_value = (
                self.fields_attrs[field_name].get('default_value', 0.0))
            values = samples.get(field_name, default_value)
            if field_name not in samples.keys() and 'infos' in samples.keys() and field_name in samples['infos'][0].keys():
                values = np.expand_dims(np.array([samples['infos'][i].get(field_name, default_value) for i in range(num_samples)]), axis=1)
            try:
                assert values.shape[0] == num_samples
                if isinstance(values[0], dict):
                    values = np.stack([np.concatenate([
                                value[key]
                                for key in value.keys()
                            ], axis=-1) for value in values])
                self.fields[field_name][index] = values
            except:
                import pdb; pdb.set_trace()
        if self.modify_rew:
            if 'reward_angle' in self.fields_attrs.keys():
                self.fields['rewards'][index] = self.fields['rewards'][index] - self.fields['reward_forward'][index] + self.fields['reward_angle'][index]
            elif 'reward_jump' in self.fields_attrs.keys():
                self.fields['rewards'][index] = self.fields['rewards'][index] + self.fields['reward_jump'][index]
            elif np.all(self.fields['rewards'][index]) <= 1 and np.all(self.fields['rewards'][index]) >= 0:
                self.fields['rewards'][index] = np.expand_dims((np.linalg.norm(self.fields['observations'][index, 3:5] - np.array([[-0.2, 0.7]]), axis=-1) <= 0.15).astype(float), axis=1)
            else:
                self.fields['rewards'][index] += 10.0 * self.fields['observations'][index, 8:9]
                # self.fields['rewards'][index] += 10.0 * self.fields['observations'][index, 9:10]

        self._advance(num_samples)

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        if self.obs_filter:
            if not hasattr(self, "filtered_indices"):
                print('Construct filtered indices for the pool...')
                self.filtered_indices = []
                for i in range(self.fields['observations'].shape[0]):
                    # if self.fields['observations'][i][0] < -3.0 or self.fields['observations'][i][0] > 3.0:
                    if self.fields['observations'][i][0] < 3.0:
                        self.filtered_indices.append(i)
            return np.random.choice(self.filtered_indices, size=batch_size)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, last_n), self._pointer
        ) % self._max_size
        return self.batch_by_indices(
            last_n_indices, field_name_filter=field_name_filter, **kwargs)

    def filter_fields(self, field_names, field_name_filter):
        if isinstance(field_name_filter, str):
            field_name_filter = [field_name_filter]

        if isinstance(field_name_filter, (list, tuple)):
            field_name_list = field_name_filter

            def filter_fn(field_name):
                return field_name in field_name_list

        else:
            filter_fn = field_name_filter

        filtered_field_names = [
            field_name for field_name in field_names
            if filter_fn(field_name)
        ]

        return filtered_field_names

    def batch_by_indices(self, indices, field_name_filter=None):
        if np.any(indices % self._max_size > self.size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        field_names = self.field_names
        if field_name_filter is not None:
            field_names = self.filter_fields(
                field_names, field_name_filter)

        return {
            field_name: self.fields[field_name][indices]
            for field_name in field_names
        }

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0

    def load_experience(self, experience_path):
        try:
            with gzip.open(experience_path, 'rb') as f:
                latest_samples = pickle.load(f)
        except:
            with open(experience_path, 'rb') as f:
                latest_samples = pickle.load(f)

        if type(latest_samples) is list:
            latest_samples = {key: np.concatenate([latest_samples[i][key] for i in range(len(latest_samples))], axis=0) for key in latest_samples[0].keys()}
        key = list(latest_samples.keys())[0]
        num_samples = latest_samples[key].shape[0]
        for field_name, data in latest_samples.items():
            assert data.shape[0] == num_samples, data.shape

        self.add_samples(latest_samples)
        self._samples_since_save = 0

    def return_all_samples(self):
        if self.obs_filter:
            if not hasattr(self, "filtered_indices"):
                print('Construct filtered indices for the pool...')
                self.filtered_indices = []
                for i in range(self.size):
                    # if self.fields['observations'][i][0] < -3.0 or self.fields['observations'][i][0] > 3.0:
                    if self.fields['observations'][i][0] < 3.0:
                        self.filtered_indices.append(i)
            filtered_fields = {
                field_name: self.fields[field_name][self.filtered_indices]
                for field_name in self.field_names
            }
            return filtered_fields
        return {
            field_name: self.fields[field_name][:self.size]
            for field_name in self.field_names
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state['fields'] = {
            field_name: self.fields[field_name][:self.size]
            for field_name in self.field_names
        }

        return state

    def __setstate__(self, state):
        if state['_size'] < state['_max_size']:
            pad_size = state['_max_size'] - state['_size']
            for field_name in state['fields'].keys():
                field_shape = state['fields_attrs'][field_name]['shape']
                state['fields'][field_name] = np.concatenate((
                    state['fields'][field_name],
                    np.zeros((pad_size, *field_shape))
                ), axis=0)

        self.__dict__ = state
