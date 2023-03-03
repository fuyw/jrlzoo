#!/bin/bash

# Script to reproduce results
envs=(
    # "humanoid-run"
    # "cheetah-run"
    # "quadruped-run"
    # "hopper-hop"
    # "cartpole-balance"
    "HalfCheetah-v4"
    "Hopper-v4"
    "Walker2d-v4"
    "Ant-v4"
)
for seed in 0 1 2 3 4
do 
    for env in ${envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$seed 
    done
done
