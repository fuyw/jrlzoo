import torch
import torch.nn as nn
import torch.nn.functional as F


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
