# Soft Actor-Critic (SAC)

A Jax implemenation of SAC for continuous control tasks.

## Baselines

Performance on some benchmark tasks: the average of the last 10 evaluation scores across 5 random seeds.

### MuJoCo


|     Env Name    |     FPS     |  Reward  |
|-----------------|-------------|----------|
|  HalfCheetah-v4 |  980~1050   |  10894.4 |
|  Hopper-v4      |  990~1010   |   3313.9 |
|  Walker2d-v4    |  920~960    |   4356.5 |
|  Ant-v4         |  770~840    |   4727.8 |

![](imgs/mujoco.png)

### DeepMind Control Suite

|     Env Name    |     FPS     |  Reward  |
|-----------------|-------------|----------|
|  cheetah-run    |   920~970   |   820.6  |
|  quadruped-run  |   730~760   |   803.5  | 
|  humanoid-run   |   640~690   |   143.3  |
|  hopper-hop     |   860~920   |   144.6  |

![](imgs/dmc.png)
