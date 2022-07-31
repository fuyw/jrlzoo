import functools

from flax import linen as nn
from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax


class QNet(nn.Module):
    act_dim: int

    def setup(self):
        self.l1 = nn.Dense(32, name="fc1")
        self.l2 = nn.Dense(3, name="fc2")
        self.l3 = nn.Dense(self.act_dim, name="fc3")

    def __call__(self, inputs: jnp.ndarray):
        x = nn.relu(self.l1(inputs))
        x = nn.relu(self.l2(x))
        q_values = self.l3(x)
        return q_values

    def Repr(self, inputs: jnp.ndarray):
        x = nn.relu(self.l1(inputs))
        x = nn.relu(self.l2(x))
        q_values = self.l3(x)
        return x, q_values


class DQN:
    def __init__(self, obs_dim, act_dim, learning_rate, gamma, seed, target_update_period):
        
        self.gamma = gamma
        self.target_update_period = target_update_period

        # Dummy Inputs
        dummy_obs = jnp.ones([1, obs_dim], dtype=jnp.float32)

        self.qnet = QNet(act_dim)
        self.pred = jax.jit(self.qnet.apply)
        params = self.qnet.init(jax.random.PRNGKey(seed), dummy_obs)["params"]
        self.target_params = params
        self.state = train_state.TrainState.create(
            apply_fn=QNet.apply,
            params=params,
            tx=optax.adam(learning_rate=learning_rate))
        self.update_step = 0

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_step(self, batch, state, target_params):
        next_q = jnp.max(self.pred({"params": target_params}, batch.next_observations),
                         axis=-1, keepdims=True)
        target_q = batch.rewards + self.gamma * batch.discounts * next_q

        def q_learning(qs, action, target_q):
            return (qs[action] - target_q)**2, qs[action]

        def loss_fn(params, batch):
            qs = self.qnet.apply({"params": params}, batch.observations)
            q_loss, q = jax.vmap(q_learning)(qs, batch.actions, target_q)
            q_loss = q_loss.mean()
            return q_loss, {"q_loss": q_loss, "q": q.mean()}
        
        (q_loss, log_info), q_grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
        state = state.apply_gradients(grads=q_grads)
        return log_info, state

    def select_action(self, params, observations):
        observations = jax.device_put(observations[None, ...])
        q_values = self.pred({"params": params}, observations)
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
        return int(action)

    def train(self, replay_buffer, batch_size):
        self.update_step += 1
        batch = replay_buffer.sample(batch_size)
        log_info, self.state = self.train_step(batch, self.state, self.target_params)

        # update the taret network
        if self.update_step % self.target_update_period:
            self.target_params = self.state.params
    
        return log_info

    def save(self, filename):
        model_file = filename + "_qnet.ckpt"
        with open(model_file, "wb") as f:
            f.write(serialization.to_bytes(self.state.params))

    def load(self, filename):
        # TODO: model loading is untested
        critic_file = filename + '_critic.ckpt'
        with open(critic_file, 'rb') as f:
            self.critic_params = serialization.from_bytes(
                self.critic_params, f.read())
        self.critic_target_params = self.critic_params
