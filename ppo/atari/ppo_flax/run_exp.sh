#!/bin/bash
# Script to reproduce results
envs=(
    "PongNoFrameskip-v4"
    "BreakoutNoFrameskip-v4"
)
for seed in 0
do
    for num in 10
    do
        for env in ${envs[*]}
        do
            python main.py \
            --config=configs/atari.py \
            --config.env_name=$env \
            --config.seed=$seed \
            --config.num_agents=$num
        done
    done
done