#!/bin/bash
atari_envs=(
    "Breakout"
    # "Asterix"
    # "BeamRider"
    # "Pong"
    # "Seaquest"
    # "SpaceInvaders"
)
for i in 10
do
    for env in ${atari_envs[*]}
    do
        python main.py \
        --config=configs/done.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done