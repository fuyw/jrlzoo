#!/bin/bash
sleep 3h
# Script to reproduce results
envs=(
    "BreakoutNoFrameskip-v4"
)
for i in 0 1 2
do
    for env in ${envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done