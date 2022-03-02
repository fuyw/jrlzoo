#!/bin/bash

# Script to reproduce results

envs=(
    "walker2d-medium-v2"
    #"HalfCheetah-v2"
    #"Walker2d-v2"
    # "Hopper-v2"
    # "Ant-v2"
)

for ((i=1;i<10;i+=1))
do 
    for env in ${envs[*]}
    do
        python main.py \
        --env_name $env \
        --seed $i
    done
done

# for ((i=0;i<3;i+=1))
# do 
#     python main.py \
#     --env ${envs[$i]} \
#     --seed ${seeds[$i]}
# done
