import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


INPUT_SIZE = 44 # state size
OUTPUT_LAYER_SIZE = 1 # value of the state

class CriticModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic():
    def __init__(self, input_dim, output_dim):
        self.model = CriticModel(input_dim, output_dim)
        self.optimizer = Adam(self.model.parameters(), lr=0.0003)

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()