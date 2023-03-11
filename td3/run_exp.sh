#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    # "cheetah-run"
    # "humanoid-run"
    # "quadruped-run"
    # "hopper-hop"

    "HalfCheetah-v2"
    "Hopper-v2"
    "Walker2d-v2"
    "Ant-v2"
)
for i in 0 1 2 3 4
do
    for env in ${mujoco_envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done    
