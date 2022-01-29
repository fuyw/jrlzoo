#!/bin/bash

# Script to reproduce results
envs=(
    "HalfCheetah-v2"
    "Hopper-v2"
    "Walker2d-v2"
    "Ant-v2"
)
for ((i=1;i<5;i+=1))
do 
    for env in ${envs[*]}
    do
        python main.py \
        --env $env \
        --seed $i
    done
done
