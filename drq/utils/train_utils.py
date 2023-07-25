import logging
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def init_fn(initializer: str, gain: float = jnp.sqrt(2)):
    if initializer == "orthogonal":
        return nn.initializers.orthogonal(gain)
    elif initializer == "glorot_uniform":
        return nn.initializers.glorot_uniform()
    elif initializer == "glorot_normal":
        return nn.initializers.glorot_normal()
    return nn.initializers.lecun_normal()


def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def target_update(params, target_params, tau: float = 0.005):
    def _update(param: FrozenDict, target_param: FrozenDict):
        return tau * param + (1 - tau) * target_param
    updated_params = jax.tree_util.tree_map(_update, params, target_params)
    return updated_params
