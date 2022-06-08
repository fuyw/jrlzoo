# Soft Actor-Critic (SAC)

A Jax implemenation of SAC for continuous control tasks.

## Baselines

Performance on some benchmark tasks: the average of the last 10 evaluation scores.


|     Env Name    |     FPS     |  Reward  |
|-----------------|-------------|----------|
|  HalfCheetah-v2 |  1020~1050  |  14591   |
|  Hopper-v2      |  990~1010   |   3000   |
|  Walker2d-v2    |  960~980    |   4951   |
|  Ant-v2         |  730~750    |   5574   |

![](imgs/sac.png)
