import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


HIDDEN_LAYER_SIZE = 32
INPUT_SIZE = 44 # state size
OUTPUT_LAYER_SIZE = 17 # probability for each action (num of actions)

class ActorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)

    def forward(self, x):
        x = F.Relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Actor():
    def __init__(self):
        self.model = ActorModel()

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        dist = Categorical(self.model.forward(state))
        return dist.sample().item()

    def update_actor(self):
        pass