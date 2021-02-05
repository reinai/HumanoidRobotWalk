import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


INPUT_SIZE = 44 # state size
OUTPUT_LAYER_SIZE = 17 # probability for each action (num of actions)

class ActorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Actor():
    def __init__(self):
        self.model = ActorModel()

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
