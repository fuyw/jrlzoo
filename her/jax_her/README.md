# Goal-Conditioned RL

## Environment Setup

Create conda environment

```shell
conda create --name jax_her python==3.10
conda activate jax_her

conda install cuda-cudart cuda-version=12

export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_opal_cuda_support=true


conda install -c conda-forge mpi4py openmpi
conda install -c conda-forge ucx

# conda install gxx_linux-64
pip install -r requirements.txt
pip install -U "jax[cuda12]"
```

## Run Experiments

### Fetch Env

```shell
mpirun -np 4 python -u main.py

python main.py
```
