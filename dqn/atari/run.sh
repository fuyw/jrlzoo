#!/bin/bash
atari_envs=(
    "Breakout"
    "Asterix"
    # "BeamRider"
    # "Pong"
    # "Seaquest"
    # "SpaceInvaders"
)
for i in 0 1 2 3 4
do
    for env in ${atari_envs[*]}
    do
        python main.py \
        --config=configs/qdagger.py \
        --config.env_name=$env \
        --config.seed=$i
    done
done