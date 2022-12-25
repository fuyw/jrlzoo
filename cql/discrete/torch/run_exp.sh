#!/bin/bash

# Script to reproduce results
# rm imgs/online_cql/*

# run offline cql agent
# python main.py --agent=cql --cql_alpha=3.0 --plot_traj

for epsilon in 0.05  0.1 0.2
do
    rm -rf imgs/online_cql
    python online_cql.py \
        --agent=cql \
        --cql_alpha=3.0 \
        --epsilon=$epsilon \
        --plot_traj
done

# python online_cql.py --agent=dqn --cql_alpha=0.5 --plot_traj

# python main.py --agent=dqn --plot_traj
