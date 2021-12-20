import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, act_dim)
    
    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        q_values = self.l3(x)
        return q_values


class DQN:
    def __init__(self, obs_dim, act_dim, learning_rate=1e-3, gamma=0.99, seed=0, target_update_period=4):
        self.gamma = gamma
        self.target_upadte_period = target_update_period

        self.qnet = QNet(obs_dim, act_dim)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=learning_rate)
