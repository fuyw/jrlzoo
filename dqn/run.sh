#!/bin/bash
sleep 3.5h
atari_envs=(
    "Pong"
    "Seaquest"
    "SpaceInvaders"
    "BeamRider"
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