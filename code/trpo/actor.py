import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.distributions import MultivariateNormal
import numpy as np


class ActorModel(nn.Module):
    """ Actor model used for policy function approximation """

    def __init__(self, input_dim, output_dim):
        """
        Intialize the actor model

        :param input_dim: dimension of input layer of nn (observation space number)
        :param output_dim: dimension of output layer of nn (action space number)
        """

        super(ActorModel, self).__init__()

        self.first_layer = nn.Linear(input_dim, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass through actor neural network

        :param x: state
        :return: action
        """

        warnings.filterwarnings("ignore")
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.third_layer(x)
        return x


class Actor():
    """ Main class that is used as Actor in TRPO algorithm """

    def __init__(self, input_dim, output_dim):
        """
        Intialize the actor class

        :param input_dim: dimension of input layer of nn (observation space number)
        :param output_dim: dimension of output layer of nn (action space number)
        """

        self.model = ActorModel(input_dim, output_dim)
        self.covariance_matrix = torch.diag(input=torch.full(size=(output_dim,), fill_value=0.5), diagonal=0)

    def get_action(self, state):
        """
        Getting action for current state

        :param state: current state
        :return: action and logaritmic probability of that action
        """

        mu = self.model.forward(state)
        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.covariance_matrix)
        action = multivariate_gaussian_distribution.sample()
        log_probability = multivariate_gaussian_distribution.log_prob(value=action)
        return action, log_probability

    def get_mean_std(self, states):
        """
        Based on states returns means and standard deviations

        :param states: observed states
        :return: means and standard deviations
        """

        mean = self.model.forward(states)
        std = torch.exp(nn.Parameter(torch.zeros(1, 17)).expand_as(mean))
        return mean, std

    def update_parameters(self, grads):
        """
        Manually updating parameters of actor model with gradient

        :param grads: gradient to update actor nn's parameters
        """
        n = 0
        for p in self.model.parameters():
            numel = p.numel()
            g = grads[n:n + numel].view(p.shape)
            p.data += g
            n += numel
