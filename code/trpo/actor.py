import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.distributions import MultivariateNormal
import numpy as np


class ActorModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorModel, self).__init__()
        self.hid1_size = 10 * input_dim
        self.hid3_size = 10 * output_dim
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))

        self.fc1 = nn.Linear(input_dim, self.hid1_size)
        self.fc2 = nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, output_dim)

    def forward(self, x):
        warnings.filterwarnings("ignore")
        x = torch.tensor(x, dtype=torch.float32)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor():
    def __init__(self, input_dim, output_dim):
        self.model = ActorModel(input_dim, output_dim)
        self.covariance_matrix = torch.diag(input=torch.full(size=(output_dim,), fill_value=0.5), diagonal=0)

    def get_action(self, state):
        mu = self.model.forward(state)
        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.covariance_matrix)
        action = multivariate_gaussian_distribution.sample()
        log_probability = multivariate_gaussian_distribution.log_prob(value=action)
        return action, log_probability

    def upgrade_parameters(self, grads):
        n = 0
        for p in self.model.parameters():
            numel = p.numel()
            g = grads[n:n + numel].view(p.shape)
            p.data += g
            n += numel
