import torch
from torch.distributions import MultivariateNormal
from actor_critic import ActorCritic


class ProximalPolicyOptimization(object):
    def __init__(self, environment):
        """
        Implementation of PPO algorithm based on: https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode

        :param environment: environment of interest, in our case: HumanoidPyBulletEnv-v0
        """
        self.environment = environment

        self.observation_dimensions = self.environment.observation_space.shape[0]
        self.action_dimensions = self.environment.action_space.shape[0]

        # initialize actor and critic
        self.actor = ActorCritic(input_dimensions=self.observation_dimensions, output_dimensions=self.action_dimensions)
        self.critic = ActorCritic(input_dimensions=self.observation_dimensions, output_dimensions=1)

    def train(self, K):
        # for k = 0, 1, 2, ... K do

        k = 0

        while k < K:
            pass