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

![](imgs/mujoco.png)

### DeepMind Control Suite

|     Env Name    |     FPS     |  Reward  |
|-----------------|-------------|----------|
|  cheetah-run    |   940~970   |   839    |
|  quadruped-run  |   730~750   |   773    | 
|  humanoid-run   |   660~680   |   132    |
|  hopper-hop     |   860~900   |   201    |

![](imgs/dmc.png)

## Versions

Some software versions:

- gym==0.21.0
- dm-control==1.0.10
- mujoco-py==2.1.2.14
- flax==0.6.1
- distrax==0.1.2
- optax==0.1.3
- jax==0.4.4
- jaxlib==0.4.4+cuda11.cudnn82

## Reset

```python
# physics.data.qacc_warmstart
from dm_control import suite

def create():
    env = suite.load(domain_name='cartpole', task_name='swingup', task_kwargs={'random': 32})
    state = np.array([1.3, 5.3, 0.1, 2.3])
    action = np.array([0.3])
    phys = env.physics
    env.reset()
    phys.set_state(state)
    env.step(action)
    obs1 = phys.render()
    obs2 = phys.render()
    h = lambda img : hash(img.data.tobytes())
    print('>>>> should be equal', h(obs1), h(obs2))

create()
create()
```

## Some implementation details

SAC is a well-known off-policy RL baseline.

However, in some tasks, the performance changes significantly even if we only modify one impletation detail.
Here is an incomplete summary of different implementation choices adopted from different popular open-sourced implementations, i.e., Dopamine, Acme, Rlkit, JaxRL.

- Use `log_alpha` or `alpha` in the entropy alpha loss.
- Set `0.5 * act_dim` or `act_dim` as the target entropy.
- How to compute the `log` for the Tanh normal policy.
- Network initialization: `orthogonal` or `glorut_uniform`.
- Critic loss: times `0.5`.
