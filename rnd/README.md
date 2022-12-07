# Exploration by Random Network Distillation

An exploration bonus for DRL agent ([Paper](https://arxiv.org/abs/1810.12894)).

The bonus is the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network. 

The intuition is that predictive models have low error in states similar to the ones they have been trained on. 

![RNG](https://openai.com/content/images/2018/10/nextstate-vs-rnd-stacked-5.svg)
