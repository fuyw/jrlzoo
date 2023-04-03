#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    # "cheetah-run"
    # "humanoid-run"
    # "quadruped-run"
    # "hopper-hop"

    "Hopper-v3"
    "Walker2d-v3"
    "Humanoid-v3"
    # "Ant-v3"
)

for i in 0
do
    for env in ${mujoco_envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done    
