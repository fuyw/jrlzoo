import jax.numpy as np
import jax.random as random
from jax import jit

a = random.normal(random.PRNGKey(1), (100, 20, 20, 3))
b = random.normal(random.PRNGKey(2), (200, 20, 20, 3))

@jit
def matmul(a, b):
  return np.transpose(np.matmul(np.transpose(a, axes=(1, 2, 0, 3)), np.transpose(b, axes=(1, 2, 3, 0))), axes=(2, 3, 0, 1))

@jit
def einsum(a, b):
  return np.einsum('nxyc,mxyc->nmxy', a, b, optimize=True)

np.sum(np.abs(einsum(a, b) - matmul(a, b)))

%timeit einsum(a, b).block_until_ready()

%timeit matmul(a, b).block_until_ready()


a = random.normal(random.PRNGKey(1), (7, 1, 23))
b = random.normal(random.PRNGKey(2), (7, 23, 200))

@jit
def einsum(a, b):
    return np.einsum('ij,ijk->ik', a, b)
%timeit einsum(a, b).block_until_ready()
# 16 µs ± 62.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


@jit
def matmul(a, b):
    return np.matmul(a, b).squeeze()
%timeit matmul(a, b).block_until_ready()
# 16.3 µs ± 70.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

