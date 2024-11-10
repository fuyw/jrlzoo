import torch
import torch.nn as nn
import torch.nn.functional as F


class actor(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.max_action = env_params["max_action"]
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params["action"])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


class critic(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.max_action = env_params["max_action"]
        self.fc1 = nn.Linear(env_params["obs"] + env_params["goal"] + env_params["action"], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
