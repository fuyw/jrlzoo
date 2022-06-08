#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    "Walker2d-v2"
    "HalfCheetah-v2"
    "Ant-v2"
    "Hopper-v2"
)
sleep 4h
for i in 3 4
do
    for env in ${mujoco_envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done