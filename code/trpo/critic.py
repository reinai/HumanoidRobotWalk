import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import warnings


class CriticModel(nn.Module):
    """ Critic model used for state value function approximation """

    def __init__(self, input_dim, output_dim):
        """
        Intialize the critic model

        :param input_dim: dimension of input layer of nn (observation space number)
        :param output_dim: dimension of output layer of nn (one -> value of the state)
        """

        super(CriticModel, self).__init__()

        self.first_layer = nn.Linear(input_dim, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, output_dim)


    def forward(self, x):
        """
        Forward pass through critic neural network

        :param x: state
        :return: value of the state
        """

        warnings.filterwarnings("ignore")
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.third_layer(x)
        return x


class Critic():
    """ Main class that is used as Critic in TRPO algorithm """

    def __init__(self, input_dim, output_dim, learning_rate):
        """
        Intialize the critic class

        :param input_dim: dimension of input layer of nn (observation space number)
        :param output_dim: dimension of output layer of nn (one -> value of the state)
        :param learning_rate: learning rate used for critic model neural netwoek
        """

        self.model = CriticModel(input_dim, output_dim)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)


    def update_critic(self, advantages):
        """
        Gradient descent on estimation of advantages

        :param advantages: estimation of advantages
        """

        loss = (advantages ** 2).mean()  # MSE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
