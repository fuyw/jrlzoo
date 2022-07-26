import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor_mu = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))


obs_dim, act_dim = 17, 6
agent = Agent(obs_dim, act_dim)
observations = torch.randn(16, obs_dim)
mu = agent.actor_mu(observations)
log_std = agent.actor_logstd.expand_as(mu)
dist = Normal(mu, torch.exp(log_std))

actions = dist.sample()
log_probs = dist.log_prob(actions).sum(axis=-1)


            # f"#Step {global_step}: reward = {eval_reward:.2f}\n"
            # f"\tvalue_loss={v_loss.item():.3f}, "
            # f"policy_loss={pg_loss.item():.3f}, "
            # f"entropy_loss={entropy_loss.item():.3f}\n"
            # f"\tavg_logp={newlogprob.mean().item():.3f}, "
            # f"max_old_logp={b_logprobs[mb_inds].max().item():.3f}, "
            # f"min_old_logp={b_logprobs[mb_inds].min().item():.3f}\n"
            # f"\tavg_old_logp={b_logprobs[mb_inds].mean().item():.3f}, "
            # f"max_logp={newlogprob.max().item():.3f}, "
            # f"min_logp={newlogprob.min().item():.3f}\n"
            # f"\tavg_value={newvalue.mean().item():.3f}, "
            # f"max_value={newvalue.max().item():.3f}, "
            # f"min_value={newvalue.min().item():.3f}\n"
            # f"\tavg_ratio={ratio.mean().item():.3f}, "
            # f"max_ratio={ratio.max().item():.3f}, "
            # f"min_ratio={ratio.min().item():.3f}\n"