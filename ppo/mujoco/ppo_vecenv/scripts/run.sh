#!/bin/bash

# Script to reproduce results
mujoco_envs=(
        'cheetah-run'
        'hopper-hop'
        'acrobot-swingup' 
        'hopper-stand'
        'humanoid-run'
        'humanoid-stand'
        'humanoid-walk'
        'finger-turn_hard'
        'pendulum-swingup'
        'quadruped-run'
        'quadruped-walk'
        'reacher-hard'
        'walker-run'
        'fish-swim'
        'swimmer-swimmer6'
)


export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl


for seed in 0 1 2 3 4
do
    for param in 0
    do
        for env_name in ${mujoco_envs[*]}
        do
            python main.py \
            --config=configs/dmc.py \
            --config.env_name=$env_name \
            --config.exp_name=ppo \
            --config.actor_num=10 \
            --config.rollout_len=125 \
            --config.seed=$seed
        done
    done
done