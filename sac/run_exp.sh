#!/bin/bash

# Script to reproduce results
envs=(
    # "cartpole-balance"
    # "cheetah-run"
    # "quadruped-run"
    # "hopper-hop"
    "HalfCheetah-v2"
    "Hopper-v2"
    "Walker2d-v2"
    "Ant-v2"
)

for seed in 0 1 2
do 
    for env in ${envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.updates_per_step=1 \
        --config.seed=$seed
    done
done
