# Meta-Q-Learning

## Getting Started

```
python run_script.py --env cheetah-dir --gpu_id 0 --seed 0
```

'env' can be humanoid-dir, ant-dir, cheetah-vel, cheetah-dir, ant-goal, and walker-rand-params. The code works on GPU and CPU machine. For the experiments in this paper, we used [p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/). For complete list of hyperparameters, please refer to the paper appendix.


In order to run this code, you will need to install Pytorch and MuJoCo. If you face any problem, please follow [PEARL](https://github.com/katerakelly/oyster/) steps to install.  

## New Environments
In order to run code with a new environment, you will need to first define an entry in ./configs/pearl_envs.json. Look at ./configs/abl_envs.json as a reference. In addation, you will need to add an env's code to rlkit/env/.

## Acknowledgement
- **rand_param_envs** and **rlkit** are completely based/copied on/from following repositories:
[rand_param_envs](https://github.com/dennisl88/rand_param_envs/tree/4d1529d61ca0d65ed4bd9207b108d4a4662a4da0) and
[PEARL](https://github.com/katerakelly/oyster/). Thanks to their authors to make them available.
We include them here to make it easier to run and work with this repository.
