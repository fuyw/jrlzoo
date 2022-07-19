from typing import Any, List
import jax
import jax.numpy as jnp


def jax_tree_stack(tree_list: List[Any]) -> jnp.ndarray:
    """Transform a list of tree into a tree with the leaves
    stacked and cast into JAX arrays.
    """
    return jax.tree_util.tree_map(
        lambda *x: jnp.stack(x, axis=0), *tree_list)

