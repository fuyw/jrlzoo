#!/bin/bash

seed=1
envs=(
    "hopper-medium-v2"
    "hopper-medium-expert-v2"
    "hopper-medium-replay-v2"
    "halfcheetah-medium-v2"
    "halfcheetah-medium-expert-v2"
    "halfcheetah-medium-replay-v2"
    "walker2d-medium-v2"
    "walker2d-medium-expert-v2"
    "walker2d-medium-replay-v2"
)


for env in ${envs[*]}
do
     python train_offline.py \
         --env_name=$env \
         --config=configs/mujoco_config.py \
         --seed=$seed
done

