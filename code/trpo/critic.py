import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


HIDDEN_LAYER_SIZE = 64
INPUT_SIZE = 44 # state size
OUTPUT_LAYER_SIZE = 1 # value of the state

class CriticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic():
    def __init__(self):
        self.model = CriticModel()
        self.optimizer = Adam(self.model.parameters(), lr=0.005)

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()