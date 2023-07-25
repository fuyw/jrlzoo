#!/bin/bash

# Script to reproduce results
envs=(
    "cheetah-run"
    # "acrobot-swingup"
    # "hopper-hop"
    # "humanoid-run"
    # "quadruped-run"
)


for seed in 0 1 2
do 
    for env in ${envs[*]}
    do
        python main.py \
        --config=configs/dmc.py \
        --config.env_name=$env \
        --config.seed=$seed 
    done
done
