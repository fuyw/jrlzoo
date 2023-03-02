#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    #"humanoid-run"
    #"cheetah-run"
    #"quadruped-run"
    #"hopper-hop"

    "HalfCheetah-v4"
    "Hopper-v4"
    "Walker2d-v4"
    "Ant-v4"
    # "Humanoid-v4"
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
