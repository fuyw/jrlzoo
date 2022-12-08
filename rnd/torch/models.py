import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 hid_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(obs_dim, hid_dim), nn.ReLU()]
        for _ in range(num_layers-1):
            layers.extend([nn.Linear(hid_dim, hid_dim), nn.ReLU()])
        layers.append(nn.Linear(hid_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        q = self.net(x)
        return q


class DQNAgent:
    def __init__(self,
                 lr: float = 1e-3,
                 act_dim: int = 2,
                 obs_dim: int = 2,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 num_layers: int = 2,
                 hid_dim: int = 64):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.qnet = QNetwork(obs_dim, act_dim, hid_dim, num_layers).to(device)
        self.target_qnet = copy.deepcopy(self.qnet).to(device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.update_step = 0

    def select_action(self, obs):
        Qs = self.qnet(torch.FloatTensor(obs).to(device))
        action = Qs.argmax().item()
        return action
 
    def update(self, batch):
        self.optimizer.zero_grad()
        Qs = self.qnet(batch.observations)
        Q = torch.gather(Qs, 1, batch.actions).squeeze()
        with torch.no_grad():
            next_Q = self.target_qnet(batch.next_observations).max(1)[0]
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q
        loss = F.mse_loss(Q, target_Q)
        loss.backward()
        self.optimizer.step()
        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        return {"avg_loss": loss, "avg_Q": Q.mean(), "avg_target_Q": target_Q.mean()}

    def sync_target_network(self):
        for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
            target_param.data.copy_(param.data)


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


class RNDModel:
    def __init__(self, obs_dim, hid_dim, out_dim, lr=1e-4):
        self.target = RNDModel(obs_dim, hid_dim, out_dim)
        self.model  = RNDModel(obs_dim, hid_dim, out_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_reward(self, x):
        y_true = self.target(x).detach()
        y_pred = self.model(x).detach()
        reward = torch.square(y_true - y_pred).sum()
        return reward

    def update(self, x):
        y_true = self.target(x).detach()
        y_pred = self.model(x).detach()
