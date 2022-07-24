#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    "humanoid-run"
    "cheetah-run"
    "quadruped-run"
    "hopper-hop"
    # "Walker2d-v2"
    # "HalfCheetah-v2"
    # "Ant-v2"
    # "Hopper-v2"
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