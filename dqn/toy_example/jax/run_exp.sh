#!/bin/bash
envs=(
    "MountainCar-v0"
    # "CartPole-v1"
)
algos=(
    "dqn"
    # "ddqn"
)
ers=(
    "er"
    # "per"
)


# python main.py --config.env_name=CartPole-v1 --config.algo=ddqn --config.er=per
for ((i=0; i<1; i+=1))
do
    for env in ${envs[*]}
    do
        for algo in ${algos[*]}
        do
            for er in ${ers[*]}
            do
                python main.py \
                --config=configs/default.py \
                --config.env_name=$env \
                --config.algo=$algo \
                --config.seed=$i \
                --config.er=$er
            done
        done
    done
done
