import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import sys
import gym
from absl import flags
from ml_collections import config_flags
import env_utils
from models import PPOAgent
from utils import ExpTuple, PPOBuffer


def run():
    config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    config = FLAGS.config
    ckpt_dir = "saved_models/HalfCheetah-v2/ppo_s0_a4_20220725_182912"
    train_envs = gym.vector.SyncVectorEnv([
        env_utils.create_env(config.env_name, seed=i) for i in range(config.actor_num)])
    eval_env = gym.make("HalfCheetah-v2")
    act_dim = eval_env.action_space.shape[0]
    obs_dim = eval_env.observation_space.shape[0]
    agent = PPOAgent(config, obs_dim=obs_dim, act_dim=act_dim, lr=1e-3)
    agent.load(ckpt_dir, 2)

    buffer = PPOBuffer(obs_dim, act_dim, config.rollout_len, config.actor_num,
                    config.gamma, config.lmbda)
    observations = train_envs.reset()  # (4, 17)
    all_experiences = []
    for _ in range(126):
        actions, log_probs = agent.sample_actions(observations)
        values = agent.get_values(observations)
        next_observations, rewards, dones, _ = train_envs.step(actions)
        experiences = [
            ExpTuple(observations[i], actions[i], rewards[i], values[i],
                    log_probs[i], dones[i]) for i in range(config.actor_num)
        ]
        all_experiences.append(experiences)
        observations = next_observations
    buffer.add_experiences(all_experiences)

    print(f"buffer.observations={buffer.observations}")
    print(f"buffer.actions={buffer.actions}")
    print(f"buffer.discounts={buffer.discounts}")
    print(f"buffer.log_probs={buffer.log_probs}")
    print(f"buffer.values={buffer.values}")
    print(f"buffer.rewards={buffer.rewards}")
    
    trajectory_batch = buffer.process_experience()


def check_dist():
    import distrax
    import jax
    import jax.numpy as jnp
    rng = jax.random.PRNGKey(0)
    rng, k1, k2 = jax.random.split(rng, 3)
    mu = jax.random.normal(k1, shape=(5, 6))
    log_std = jax.random.normal(k2, shape=(6,))
    std = jnp.exp(log_std)
    # dist = distrax.Normal(mu, std)
    dist = distrax.MultivariateNormalDiag(mu, std)
    actions, log_probs = dist.sample_and_log_prob(seed=rng)
    dist.log_prob(actions)


def test_ac():
    class ActorCritic(nn.Module):
        act_dim: int
        hidden_dims: Tuple[int] = (64, 64)
        initializer: str = "orthogonal"

        def setup(self):
            self.actor = Actor(self.act_dim, hidden_dims=self.hidden_dims)
            self.critic = Critic(self.hidden_dims, self.initializer)

        def __call__(self, key: Any, observations: jnp.ndarray):
            mean_actions, sampled_actions, log_probs = self.actor(
                key, observations)
            values = self.critic(observations)
            return mean_actions, sampled_actions, log_probs, values

        def get_value(self, observations: jnp.ndarray) -> jnp.ndarray:
            values = self.critic(observations)
            return values

        def get_logp(self, observations: jnp.ndarray,
                    actions: jnp.ndarray) -> jnp.ndarray:
            log_probs, entropy = self.actor.get_logp(observations, actions)
            values = self.critic(observations)
            return log_probs, values, entropy

