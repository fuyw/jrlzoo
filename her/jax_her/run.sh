#!/bin/bash


# Script to reproduce results
envs=(
    # "FetchReach-v3"
    # "FetchPush-v3"
    "FetchPickAndPlace-v3"
)


for seed in 0
do
    for env in ${envs[*]}
    do
        python main.py \
        --config.env_name=$env \
        --config.seed=$seed \
        --config.model=sac \
        --config.max_episode_steps=100 \
        --config.num_envs=16
    done
done
