#!/bin/bash

# Script to reproduce results
mujoco_envs=(
    "halfcheetah-medium-v2"
    "halfcheetah-random-v2"
    "halfcheetah-medium-replay-v2"

    "hopper-medium-v2"
    "hopper-random-v2"
    "hopper-medium-replay-v2"

    "walker2d-medium-v2"
    "walker2d-random-v2"
    "walker2d-medium-replay-v2"
)


antmaze_envs=(
    "antmaze-large-play-v0"
    "antmaze-large-diverse-v0"
    "antmaze-medium-play-v0"
    "antmaze-medium-diverse-v0"
    "antmaze-umaze-v0"
    "antmaze-umaze-diverse-v0"
)

for ((i=0;i<5;i+=1))
do
    for env in ${mujoco_envs[*]}
    do
        python main.py \
        --config=configs/mujoco.py \
        --config.env_name=$env \
        --config.seed=$i \
        --config.initializer=orthogonal
    done

    for env in ${antmaze_envs[*]}
    do
        python main.py \
        --config=configs/antmaze.py \
        --config.env_name=$env \
        --config.seed=$i \
        --config.initializer=orthogonal
    done
done