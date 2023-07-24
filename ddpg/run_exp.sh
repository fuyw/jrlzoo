#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    "HalfCheetah-v3"
    "Hopper-v3"
    "Walker2d-v3"
    "Ant-v3"
)


for i in 1 2
do
    for env in ${mujoco_envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done
