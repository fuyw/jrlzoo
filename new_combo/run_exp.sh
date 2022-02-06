#!/bin/bash

# Script to reproduce results
envs=(
    "halfcheetah-medium-v2"
    "hopper-medium-v2"
    "walker2d-medium-v2"
)

for min_q_weight in 0.5 3 5
do 
    for env in ${envs[*]}
    do
        for real_ratio in 0.5 0.75 0.99
        do
            python main.py \
            --env $env \
            --real_ratio $real_ratio \
            --min_q_weight $min_q_weight
        done
    done
done
