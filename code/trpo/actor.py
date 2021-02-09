import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hid1_size = 10 * input_dim
        self.hid3_size = 10 * output_dim
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))

        self.fc1 = nn.Linear(input_dim, self.hid1_size)
        self.fc2 = nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)


class Actor():
    def __init__(self, input_dim, output_dim):
        self.model = ActorModel(input_dim, output_dim)

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0)
        dist = Categorical(torch.distributions.utils.clamp_probs(self.model.forward(state)))
        return dist.sample().item()

    def upgrade_parameters(self, grads):
        n = 0
        for p in self.model.parameters():
            numel = p.numel()
            g = grads[n:n + numel].view(p.shape)
            p.data += g
            n += numel
