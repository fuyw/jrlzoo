# JaxTD3

A Jax implementation of TD3.

![res](imgs/res.png)

To run experiments in `mujoco` environment
```
python main.py --config=configs/mujoco.py --config.env_name=halfcheetah-medium-v2 --config.seed=0
```

To run experiments in `antmaze` environment
```
python main.py --config=configs/antmaze.py --config.env_name=antmaze-medium-play-v0 --config.seed=0
```
