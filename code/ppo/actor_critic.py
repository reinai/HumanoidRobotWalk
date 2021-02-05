import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        """
        Actor-Critic architecture is used to explicitly represent the policy that is independent of the value function.
        The policy structure is known as the actor, because it is used to select actions, and the estimated value
        function is known as the critic, because it criticizes the actions made by the actor.

        :param input_dimensions: take in the observation Box dimensions
        :param output_dimensions: dimensions for distribution over actions (for actor),
                                  value for state = dimension is 1 (for critic)
        """
        super(ActorCritic, self).__init__()

        # 64 is chosen based on Stanford's recommendation, experimentally backed up
        self.first_layer = nn.Linear(input_dimensions, 64)
        self.second_layer = nn.Linear(64, 64)
        self.third_layer = nn.Linear(64, output_dimensions)

    def forward(self, observation):
        # We use Box array which deals with real valued quantities
        observation = torch.tensor(observation, dtype=torch.float32)

        first_activation_function = F.relu(self.first_layer(observation))
        second_activation_function = F.relu(self.second_layer(first_activation_function))

        output = self.third_layer(second_activation_function)

        return output
