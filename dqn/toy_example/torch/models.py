import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, obs_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class DQNAgent:
    def __init__(self,
                 lr: float = 1e-3,
                 obs_dim: int = 2,
                 act_dim: int = 2,
                 hid_dim: int = 64,
                 gamma: float = 0.99,
                 tau: float = 0.005):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.qnet = QNetwork(obs_dim, act_dim, hid_dim).to(device)
        self.target_qnet = copy.deepcopy(self.qnet).to(device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)

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

    def update(self, batch):
        self.optimizer.zero_grad()
        Qs = self.qnet(batch.observations)                # (256, act_dim)
        Q = torch.gather(Qs, 1, batch.actions).squeeze()  # (256,)
        with torch.no_grad():
            next_Q = self.target_qnet(batch.next_observations).max(1)[0]  # (256,)
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q  # (256,)
        loss = F.mse_loss(Q, target_Q)
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        return {"avg_loss": loss, "avg_Q": Q.mean(), "avg_target_Q": target_Q.mean()}

    def save(self, fname):
        torch.save(self.qnet.state_dict(), f"{fname}_qnet")
        torch.save(self.target_qnet.state_dict(), f"{fname}_target_qnet")

    def load(self, fname):
        self.qnet.load_state_dict(torch.load(f"{fname}_qnet"))
        self.target_qnet.load_state_dict(torch.load(f"{fname}_target_qnet"))
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
