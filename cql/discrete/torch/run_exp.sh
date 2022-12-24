#!/bin/bash

# Script to reproduce results
for agent in cql
do 
    for cql_alpha in 0.5 1.0 3.0 5.0 10.0
    do
        python main.py \
        --agent=$agent \
        --cql_alpha=$cql_alpha
    done
done
