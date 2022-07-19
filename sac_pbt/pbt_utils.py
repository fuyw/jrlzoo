from typing import Any, Callable, NamedTuple, Tuple
import gym
import haiku
import numpy as np
import jax
import jax.numpy as jnp
from networks import LayerNormMLP, NearZeroInitializedLinear, TanhToSpec


class TD3Networks(NamedTuple):
    policy_network: Any
    critic_network: Any
    twin_critic_network: Any

    init_policy_network: Any
    init_critic_network: Any
    init_twin_critic_network: Any

    add_policy_noise: Any


def make_default_networks(
    gym_env: gym.Env,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
) -> TD3Networks:

    action_shape = gym_env.action_space.shape
    action_min = gym_env.action_space.low.astype(np.float32)
    action_max = gym_env.action_space.high.astype(np.float32)

    def add_policy_noise(
        action: jnp.ndarray,
        key: jax.random.PRNGKey,
        target_sigma: float,
        noise_clip: float,
    ) -> jnp.ndarray:
        """Adds action noise to bootstrapped Q-value estimate in critic loss."""
        noise = jax.random.normal(key=key, shape=action.shape) * target_sigma
        noise = jnp.clip(noise, -noise_clip, noise_clip)
        return jnp.clip(action + noise, action_min, action_max)

    def _policy_forward_pass(obs: jnp.ndarray) -> jnp.ndarray:
        policy_network = haiku.Sequential(
            [
                LayerNormMLP(hidden_layer_sizes, activate_final=True),
                NearZeroInitializedLinear(np.prod(action_shape, dtype=int)),
                TanhToSpec(min_value=action_min, max_value=action_max),
            ]
        )
        return policy_network(obs)

    def _critic_forward_pass(obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        critic_network = haiku.Sequential(
            [
                LayerNormMLP(list(hidden_layer_sizes) + [1]),
            ]
        )
        input_ = jnp.concatenate([obs, action], axis=-1)
        value = critic_network(input_)
        return jnp.squeeze(value)

    policy = haiku.without_apply_rng(haiku.transform(_policy_forward_pass))
    critic = haiku.without_apply_rng(haiku.transform(_critic_forward_pass))

    # Create dummy observations and actions to create network parameters.
    dummy_action = jnp.expand_dims(
        jnp.zeros_like(gym_env.action_space.sample().astype(np.float32)), axis=0
    )
    dummy_obs = jax.tree_map(
        lambda x: jnp.expand_dims(jnp.zeros_like(x.astype(np.float32)), axis=0),
        gym_env.observation_space.sample(),
    )

    return TD3Networks(
        policy_network=policy.apply,
        critic_network=critic.apply,
        twin_critic_network=critic.apply,
        add_policy_noise=add_policy_noise,
        init_policy_network=lambda key: policy.init(key, dummy_obs),
        init_critic_network=lambda key: critic.init(key, dummy_obs, dummy_action),
        init_twin_critic_network=lambda key: critic.init(key, dummy_obs, dummy_action),
    )


