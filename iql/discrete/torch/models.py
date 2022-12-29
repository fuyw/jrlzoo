import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 hid_dim: int,
                 out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hid_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.logits = MLP(obs_dim, hid_dim, act_dim)
    
    def forward(self, x):
        logits = self.logits(x)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution


class QNetwork(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hid_dim: int = 128):
        super().__init__()
        self.net = MLP(obs_dim, hid_dim, act_dim)

    def forward(self, x):
        q = self.net(x)
        return q


class IQLAgent:
    def __init__(self,
                 lr: float = 1e-3,
                 obs_dim: int = 2,
                 act_dim: int = 2,
                 hid_dim: int = 64,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.7,
                 **kwargs):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.expectile = expectile

        self.qnet = QNetwork(obs_dim, act_dim, hid_dim).to(device)
        self.target_qnet = copy.deepcopy(self.qnet).to(device)
        self.vnet = QNetwork(obs_dim, 1, hid_dim).to(device)
        self.actor = Actor(obs_dim, act_dim, hid_dim).to(device)

        self.q_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.v_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=self.lr)
        self.a_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def select_action(self, obs):
        with torch.no_grad():
            Qs = self.qnet(torch.FloatTensor(obs).to(device))
        action = Qs.argmax().item()
        return action

    def get_qvalues(self, batch):
        with torch.no_grad():
            Qs = self.qnet(batch.observations)
        Q = torch.gather(Qs, 1, batch.actions).squeeze()  # (256,)
        return Q.mean().item()

    def update_v(self, batch):
        with torch.no_grad():
            Qs = self.target_qnet(batch.observations)
        Q = torch.gather(Qs, 1, batch.actions)

        self.v_optimizer.zero_grad()
        V = self.vnet(batch.observations)
        weight = torch.abs(self.expecitle - (Q-V<0).float())
        v_loss = (weight * torch.square(Q - V)).mean()
        v_loss.backward()
        self.v_optimizer.step()
        return {"v_loss": v_loss}

    def update_q(self, batch):   
        with torch.no_grad():
            next_Q = self.vnet(batch.next_observations)[0]
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q

        self.optimizer.zero_grad()
        Qs = self.qnet(batch.observations)  # (256, act_dim)
        Q = torch.gather(Qs, 1, batch.actions).squeeze()  # (256, 1)
        loss = F.mse_loss(Q, target_Q)
        loss.backward()
        self.optimizer.step()

        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        return {"avg_loss": loss, "avg_Q": Q.mean(), "avg_target_Q": target_Q.mean()}

    def update_pi(self, batch):
        with torch.no_grad():
            V = self.vnet(batch.observations)
            Qs = self.target_qnet(batch.observations)
        Q = torch.gather(Qs, 1, batch.actions)
        advantage = Q - V

        self.a_optimizer.zero_grad()
        action_distribution = self.actor(batch.observations)
        logp = action_distribution.log_prob(batch.actions)
        a_loss = (-advantage * logp).mean()
        a_loss.backward()
        self.a_optimizer.step()
        return {"a_loss": a_loss}

    def update(self, batch):
        v_log_info = self.update_v(batch)
        q_log_info = self.update_q(batch)
        a_log_info = self.update_a(batch)
        return {**v_log_info, **q_log_info, **a_log_info}

    def save(self, fname):
        torch.save(self.actor.state_dict(), f"{fname}_actor")
        torch.save(self.vnet.state_dict(), f"{fname}_vnet")
        torch.save(self.qnet.state_dict(), f"{fname}_qnet")
        torch.save(self.target_qnet.state_dict(), f"{fname}_target_qnet")

    def load(self, fname):
        self.actor.load_state_dict(torch.load(f"{fname}_actor"))
        self.vnet.load_state_dict(torch.load(f"{fname}_vnet"))
        self.qnet.load_state_dict(torch.load(f"{fname}_qnet"))
        self.target_qnet.load_state_dict(torch.load(f"{fname}_target_qnet"))
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
