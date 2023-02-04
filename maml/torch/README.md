# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)

## Usage

#### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments with MAML. Note that by default, logs are available in [`train.py`](train.py) but **are not** saved (eg. the returns during meta-training). For example, to run the script on HalfCheetah-Vel:
```
python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder maml-halfcheetah-vel --seed 1 --num-workers 8
```

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py --config maml-halfcheetah-vel/config.json --policy maml-halfcheetah-vel/policy.th --output maml-halfcheetah-vel/results.npz --meta-batch-size 20 --num-batches 10  --num-workers 8
```

## References
This project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in Pytorch. These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep
Networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

If you want to cite this paper
```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```

If you want to cite this implementation:
```
@misc{deleu2018mamlrl,
  author = {Tristan Deleu},
  title  = {{Model-Agnostic Meta-Learning for Reinforcement Learning in PyTorch}},
  note   = {Available at: https://github.com/tristandeleu/pytorch-maml-rl},
  year   = {2018}
}
```
