#!/bin/bash

# Script to reproduce results
envs=(
    "hopper-hop"
    # "cheetah-run"
    # "humanoid-run"
    # "quadruped-run"

    # "HalfCheetah-v2"
    # "Hopper-v2"
    # "Walker2d-v2"
    # "Ant-v2"
)

for seed in 0
do 
    for env in ${envs[*]}
    do
        python main.py \
        --config=configs/dmc.py \
        --config.env_name=$env \
        --config.seed=$seed 
    done
done
