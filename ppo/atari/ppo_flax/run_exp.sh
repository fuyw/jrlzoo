#!/bin/bash
# Script to reproduce results
envs=(
    "BreakoutNoFrameskip-v4"
    # "PongNoFrameskip-v4"
)
for seed in 0
do
    for num in 8
    do
        for env in ${envs[*]}
        do
            python main.py \
            --config=configs/atari.py \
            --config.env_name=$env \
            --config.seed=$seed \
            --config.actor_num=$num
        done
    done
done