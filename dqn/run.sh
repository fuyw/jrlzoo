#!/bin/bash
sleep 0.5h
# Script to reproduce results
atari_envs=(
    "Asterix"
    "Qbert"
)
for i in 1
do
    for env in ${atari_envs[*]}
    do
        python main.py \
        --config=configs/atari.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done