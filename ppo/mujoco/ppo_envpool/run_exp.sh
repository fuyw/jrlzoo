#!/bin/bash
# Script to reproduce results
envs=(
    "Breakout"
    "Pong"
)
for seed in 0
do
    for num in 5
    do
        for env in ${envs[*]}
        do
            python main.py \
            --config=configs/atari.py \
            --config.env_name=$env \
            --config.seed=$seed \
            --config.actor_num=$num \
            --config.wait_num=$num
        done
    done
done
