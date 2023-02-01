# JAX implementation of Meta-Q-Learning

## Main Ideas

- Q-learning is competitive with state-of-the-art meta-RL algorithms if given access to a context variable that is a representation of the past trajectory. 

- A multi-task objective to maximize the average reward across the training tasks is an effective method to meta-train RL policies. 

- Past data from the meta-training replay buffer can be recycled to adapt the policy on a new task using off-policy updates. 


## Acknowledgements

[Official pytorch implementation](https://github.com/amazon-science/meta-q-learning)
