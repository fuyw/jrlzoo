#!/bin/bash

# Script to reproduce results
atari_envs=(
    "Breakout"
    "Asterix"
)
for i in 0
do
    for env in ${atari_envs[*]}
    do
        python main.py \
        --config=configs/atari.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done