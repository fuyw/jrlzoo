# Offline Reinforcement Learning with Implicit Q-Learning

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

For a PyTorch reimplementation see https://github.com/rail-berkeley/rlkit/tree/master/examples/iql

## How to run the code

### Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install "jax[cuda111]<=0.21.1" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```

## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).

python -m SimpleSAC.conservative_sac_main \
    --env 'antmaze-medium-diverse-v2' \
    --cql.cql_min_q_weight=5.0 \
    --cql.cql_max_target_backup=True \
    --cql.cql_target_action_gap=0.2 \
    --orthogonal_init=True \
    --cql.cql_lagrange=True \
    --cql.cql_temp=1.0 \
    --cql.policy_lr=1e-4 \
    --cql.qf_lr=3e-4 \
    --cql.cql_clip_diff_min=-200 \
    --reward_scale=10.0 \
    --reward_bias=-5.0 \
    --policy_arch='256-256' \
    --qf_arch='256-256-256' \
    --policy_log_std_multiplier=0.0 \
    --eval_period=50 \
    --eval_n_trajs=100 \
    --n_epochs=1200 \
    --bc_epochs=40 \
    --logging.output_dir './experiment_output'
