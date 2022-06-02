#!/bin/bash

# Script to reproduce results
envs=(
    "quadruped-run"
    "hopper-hop"
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
