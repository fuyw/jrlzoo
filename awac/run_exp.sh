#!/bin/bash

# Script to reproduce results
envs=(
    #"halfcheetah-medium-replay-v2"
    #"halfcheetah-medium-expert-v2"
    "halfcheetah-medium-v2"
    #"hopper-medium-v2"
    #"walker2d-medium-v2"
    #"hopper-medium-replay-v2"
    #"hopper-medium-expert-v2"
    #"walker2d-medium-replay-v2"
    #"walker2d-medium-expert-v2"
)
for seed in 0
do 
    for env in ${envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$seed
    done
done
