#!/bin/bash

# Script to reproduce results

envs=(
    "HalfCheetah-v2"
    # "Walker2d-v2"
    # "Hopper-v2"
    # "Ant-v2"
)

seeds=(
    "8"
    "0"
    "2"
)

for ((i=0;i<3;i+=1))
do 
    for env in ${envs[*]}
    do
        python main.py \
        --env $env \
        --seed $i
    done
done

# for ((i=0;i<3;i+=1))
# do 
#     python main.py \
#     --env ${envs[$i]} \
#     --seed ${seeds[$i]}
# done
