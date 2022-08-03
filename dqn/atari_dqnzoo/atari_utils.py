from typing import Optional, Tuple

import atari_py
import dm_env
from dm_env import specs
import gym
import numpy as np

# Game: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: score human (float).
_ATARI_DATA = {
    'alien': (227.8, 7127.7),
    'amidar': (5.8, 1719.5),
    'assault': (222.4, 742.0),
    'asterix': (210.0, 8503.3),
    'asteroids': (719.1, 47388.7),
    'atlantis': (12850.0, 29028.1),
    'bank_heist': (14.2, 753.1),
    'battle_zone': (2360.0, 37187.5),
    'beam_rider': (363.9, 16926.5),
    'berzerk': (123.7, 2630.4),
    'bowling': (23.1, 160.7),
    'boxing': (0.1, 12.1),
    'breakout': (1.7, 30.5),
    'centipede': (2090.9, 12017.0),
    'chopper_command': (811.0, 7387.8),
    'crazy_climber': (10780.5, 35829.4),
    'defender': (2874.5, 18688.9),
    'demon_attack': (152.1, 1971.0),
    'double_dunk': (-18.6, -16.4),
    'enduro': (0.0, 860.5),
    'fishing_derby': (-91.7, -38.7),
    'freeway': (0.0, 29.6),
    'frostbite': (65.2, 4334.7),
    'gopher': (257.6, 2412.5),
    'gravitar': (173.0, 3351.4),
    'hero': (1027.0, 30826.4),
    'ice_hockey': (-11.2, 0.9),
    'jamesbond': (29.0, 302.8),
    'kangaroo': (52.0, 3035.0),
    'krull': (1598.0, 2665.5),
    'kung_fu_master': (258.5, 22736.3),
    'montezuma_revenge': (0.0, 4753.3),
    'ms_pacman': (307.3, 6951.6),
    'name_this_game': (2292.3, 8049.0),
    'phoenix': (761.4, 7242.6),
    'pitfall': (-229.4, 6463.7),
    'pong': (-20.7, 14.6),
    'private_eye': (24.9, 69571.3),
    'qbert': (163.9, 13455.0),
    'riverraid': (1338.5, 17118.0),
    'road_runner': (11.5, 7845.0),
    'robotank': (2.2, 11.9),
    'seaquest': (68.4, 42054.7),
    'skiing': (-17098.1, -4336.9),
    'solaris': (1236.3, 12326.7),
    'space_invaders': (148.0, 1668.7),
    'star_gunner': (664.0, 10250.0),
    'surround': (-10.0, 6.5),
    'tennis': (-23.8, -8.3),
    'time_pilot': (3568.0, 5229.2),
    'tutankham': (11.4, 167.6),
    'up_n_down': (533.4, 11693.2),
    'venture': (0.0, 1187.5),
    # Note the random agent score on Video Pinball is sometimes greater than the
    # human score under other evaluation methods.
    'video_pinball': (16256.9, 17667.9),
    'wizard_of_wor': (563.5, 4756.5),
    'yars_revenge': (3092.9, 54576.9),
    'zaxxon': (32.5, 9173.3),
}
_RANDOM_COL = 0
_HUMAN_COL = 1
ATARI_GAMES = tuple(sorted(_ATARI_DATA.keys()))
_GYM_ID_SUFFIX = '-xitari-v1'


def get_human_normalized_score(game: str, raw_score: float) -> float:
    """Converts game score to human-normalized score."""
    game_scores = _ATARI_DATA.get(game, (np.nan, np.nan))
    random, human = game_scores[_RANDOM_COL], game_scores[_HUMAN_COL]
    return (raw_score - random) / (human - random)


def _register_atari_environments():
    """Registers Atari environments in Gym to be as similar to Xitari as possible.
    Main difference from PongNoFrameSkip-v4, etc. is max_episode_steps is unset and only the usual 57 Atari games are registered.
    """
    for game in ATARI_GAMES:
        gym.envs.registration.register(
            id=game + _GYM_ID_SUFFIX,  # Add suffix so ID has required format.
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={  # Explicitly set all known arguments.
                'game': game,
                'mode': None,  # Not necessarily the same as 0.
                'difficulty': None,  # Not necessarily the same as 0.
                'obs_type': 'image',
                'frameskip': 1,  # Get every frame.
                'repeat_action_probability': 0.0,  # No sticky actions.
                'full_action_space': False,
            },
            max_episode_steps=None,  # No time limit, handled in training run loop.
            nondeterministic=False,  # Xitari is deterministic.
        )


_register_atari_environments()


class GymAtari(dm_env.Environment):
    """Gym Atari with a `dm_env.Environment` interface."""

    def __init__(self, game, seed):
        self._gym_env = gym.make(game + _GYM_ID_SUFFIX)
        self._gym_env.seed(seed)
        self._start_of_episode = True

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment and starts a new episode."""
        observation = self._gym_env.reset()
        lives = np.int32(self._gym_env.ale.lives())
        timestep = dm_env.restart((observation, lives))
        self._start_of_episode = False
        return timestep

    def step(self, action: np.int32) -> dm_env.TimeStep:
        """Updates the environment given an action and returns a timestep."""
        # If the previous timestep was LAST then we call reset() on the Gym
        # environment, otherwise step(). Although Gym environments allow you to step
        # through episode boundaries (similar to dm_env) they emit a warning.
        if self._start_of_episode:
            step_type = dm_env.StepType.FIRST
            observation = self._gym_env.reset()
            discount = None
            reward = None
            done = False
        else:
            observation, reward, done, info = self._gym_env.step(action)
            if done:
                assert 'TimeLimit.truncated' not in info, 'Should never truncate.'
                step_type = dm_env.StepType.LAST
                discount = 0.
            else:
                step_type = dm_env.StepType.MID
                discount = 1.

        lives = np.int32(self._gym_env.ale.lives())
        timestep = dm_env.TimeStep(
            step_type=step_type,
            observation=(observation, lives),
            reward=reward,
            discount=discount,
        )
        self._start_of_episode = done
        return timestep

    def observation_spec(self) -> Tuple[specs.Array, specs.Array]:
        space = self._gym_env.observation_space
        return (specs.Array(shape=space.shape, dtype=space.dtype, name='rgb'),
                specs.Array(shape=(), dtype=np.int32, name='lives'))

    def action_spec(self) -> specs.DiscreteArray:
        space = self._gym_env.action_space
        return specs.DiscreteArray(num_values=space.n,
                                   dtype=np.int32,
                                   name='action')

    def close(self):
        self._gym_env.close()


class RandomNoopsEnvironmentWrapper(dm_env.Environment):
    """Adds a random number of noop actions at the beginning of each episode."""

    def __init__(self,
                 environment: dm_env.Environment,
                 max_noop_steps: int,
                 min_noop_steps: int = 0,
                 noop_action: int = 0,
                 seed: Optional[int] = None):
        """Initializes the random noops environment wrapper."""
        self._environment = environment
        if max_noop_steps < min_noop_steps:
            raise ValueError(
                'max_noop_steps must be greater or equal min_noop_steps')
        self._min_noop_steps = min_noop_steps
        self._max_noop_steps = max_noop_steps
        self._noop_action = noop_action
        self._rng = np.random.RandomState(seed)

    def reset(self):
        """Begins new episode.

    This method resets the wrapped environment and applies a random number
    of noop actions before returning the last resulting observation
    as the first episode timestep. Intermediate timesteps emitted by the inner
    environment (including all rewards and discounts) are discarded.

    Returns:
      First episode timestep corresponding to the timestep after a random number
      of noop actions are applied to the inner environment.

    Raises:
      RuntimeError: if an episode end occurs while the inner environment
        is being stepped through with the noop action.
    """
        return self._apply_random_noops(
            initial_timestep=self._environment.reset())

    def step(self, action):
        """Steps environment given action.

    If beginning a new episode then random noops are applied as in `reset()`.

    Args:
      action: action to pass to environment conforming to action spec.

    Returns:
      `Timestep` from the inner environment unless beginning a new episode, in
      which case this is the timestep after a random number of noop actions
      are applied to the inner environment.
    """
        timestep = self._environment.step(action)
        if timestep.first():
            return self._apply_random_noops(initial_timestep=timestep)
        else:
            return timestep

    def _apply_random_noops(self, initial_timestep):
        assert initial_timestep.first()
        num_steps = self._rng.randint(self._min_noop_steps,
                                      self._max_noop_steps + 1)
        timestep = initial_timestep
        for _ in range(num_steps):
            timestep = self._environment.step(self._noop_action)
            if timestep.last():
                raise RuntimeError(
                    'Episode ended while applying %s noop actions.' %
                    num_steps)

        # We make sure to return a FIRST timestep, i.e. discard rewards & discounts.
        return dm_env.restart(timestep.observation)

    ## All methods except for reset and step redirect to the underlying env.

    def observation_spec(self):
        return self._environment.observation_spec()

    def action_spec(self):
        return self._environment.action_spec()

    def reward_spec(self):
        return self._environment.reward_spec()

    def discount_spec(self):
        return self._environment.discount_spec()

    def close(self):
        return self._environment.close()


def create_env(env_name, seed=42):
    env = GymAtari(env_name, seed=seed)
    return RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=seed
    )
