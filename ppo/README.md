# Proximal Policy Optimization (PPO)

PPO is an on-policy model-free RL algorithm, which requires a large amount of samples to achieve comparable performance w.r.t. other off-policy algorithms.
([OpenAI Spinning Up baselines](https://spinningup.openai.com/en/latest/spinningup/bench.html#halfcheetah-pytorch-versions))

![Spinning Up Baselines](https://spinningup.openai.com/en/latest/_images/pytorch_halfcheetah_performance.svg)

## PPO for Atari

- `ppo_envpool` uses the envpool.
- `ppo_flax` is adapted from the flax official example.

## PPO for Mujoco

- `poo_vecenv` uses the gym vectorized environments.