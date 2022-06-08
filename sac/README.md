# Soft Actor-Critic (SAC)

A Jax implemenation of SAC for continuous control tasks.

## Baselines

Performance on some benchmark tasks: the average of the last 10 evaluation scores across 5 random seeds.

### MuJoCo


|     Env Name    |     FPS     |  Reward  |
|-----------------|-------------|----------|
|  HalfCheetah-v2 |  1020~1050  |  14311   |
|  Hopper-v2      |  990~1010   |   2947   |
|  Walker2d-v2    |  960~980    |   5447   |
|  Ant-v2         |  730~750    |   5927   |

![](imgs/sac.png)

### DeepMind Control Suite

|     Env Name    |     FPS     |  Reward  |
|-----------------|-------------|----------|
|  cheetah-run    |  990~1010   |  14311   |
|  quadruped-run  |  730~750    |          | 
|  hopper-hop     |  930~960    |          |
