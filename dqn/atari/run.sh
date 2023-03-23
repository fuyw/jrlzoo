#!/bin/bash
atari_envs=(
    "Asterix"
    "BeamRider"
    # "Breakout"
    # "Pong"
    # "Seaquest"
    # "SpaceInvaders"
)
sleep 3h
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