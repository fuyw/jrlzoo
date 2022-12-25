#!/bin/bash

# Script to reproduce results
# rm imgs/online_cql/*

# run offline cql agent
# python main.py --agent=cql --cql_alpha=3.0 --plot_traj

for epsilon in 0.05 0.1 0.2
do
    python online.py \
        --agent=dqn \
        --epsilon=$epsilon
done

for epsilon in 0.05 0.1 0.2
do
    python online.py \
        --agent=dqn \
        --epsilon=$epsilon \
        --new_buffer
done


for alpha in 0.0 0.5 1.0 3.0
do
    for epsilon in 0.05 0.1 0.2
    do
        python online.py \
            --agent=cql \
            --cql_alpha=$alpha \
            --epsilon=$epsilon
    done
done

for alpha in 0.0 0.5 1.0 3.0
do
    for epsilon in 0.05 0.1 0.2
    do
        python online.py \
            --agent=cql \
            --cql_alpha=$alpha \
            --epsilon=$epsilon \
            --new_buffer
    done
done
