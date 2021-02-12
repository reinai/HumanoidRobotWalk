import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class CriticModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CriticModel, self).__init__()
        self.hid1_size = 10 * input_dim
        self.hid3_size = 5
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))

        self.fc1 = nn.Linear(input_dim, self.hid1_size)
        self.fc2 = nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, output_dim)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class Critic():
    def __init__(self, input_dim, output_dim):
        self.model = CriticModel(input_dim, output_dim)
        self.optimizer = Adam(self.model.parameters(), lr=1e-2 / np.sqrt(self.model.hid2_size))

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
