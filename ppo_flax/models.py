from flax import linen as nn
import distrax
import multiprocessing as mp
import jax.numpy as jnp
import env_utils


class ActorCritic(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name="conv1", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name="conv2", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name="conv3", dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        x = nn.Dense(features=512, name="hidden", dtype=jnp.float32)(x)
        x = nn.relu(x)

        value = nn.Dense(features=1, name="value", dtype=jnp.float32)(x)
        logits = nn.Dense(features=self.act_dim, name="logits", dtype=jnp.float32)(x)
        # policy_log_probabilities = nn.log_softmax(logits)
        # return policy_log_probabilities, value
        action_distribution = distrax.Categorical(logits=logits)
        return action_distribution, value.squeeze(-1)


class RemoteSimulator:
    """Remote actor in a separate process."""

    def __init__(self, env_name: str, rank: int):
        """Start the remote process and create Pipe() to communicate with it."""
        parent_conn, child_conn = mp.Pipe()
        self.proc = mp.Process(target=self.rcv_action_send_exp,
                               args=(child_conn, env_name, rank))
        self.proc.daemon = True
        self.conn = parent_conn
        self.proc.start()

    def rcv_action_send_exp(self, conn, env_name: str, rank: int = 0):
        """Run remote actor.

        Receive action from the main learner, perform one step of simulation and
        send back collected experience.
        """
        env = env_utils.create_env(env_name, clip_rewards=True, seed=rank+100)
        while True:
            obs, done = env.reset(), False
            while not done:
                # (1) send observation to the learner
                conn.send(obs[None, ...])

                # (2) receive sampled action from the learner
                action = conn.recv()

                # (3) interact with the environment
                next_obs, reward, done, _ = env.step(action)
                experience = (obs, action, reward, done)

                # (4) send next observation to the learner
                conn.send(experience)
                if done:
                    break
                obs = next_obs
