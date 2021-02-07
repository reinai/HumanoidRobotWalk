import torch
from actor_critic import ActorCritic
from torch.optim import Adam as AdamOptimizer
from anemic_domain_models import BatchData
from torch.distributions import MultivariateNormal
import numpy as np
import torch.nn as nn


class ProximalPolicyOptimization(object):
    def __init__(self, environment, **hyper_parameters):
        """
        Implementation of PPO algorithm based on: https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode

        :param environment: environment of interest, in our case: HumanoidPyBulletEnv-v0
        :param hyper_parameters: PPO algorithm hyper-parameters
        """
        self.__init__hyper_parameters(hyper_parameters=hyper_parameters)

        self.environment = environment

        self.observation_dimensions = self.environment.observation_space.shape[0]  # 44
        self.action_dimensions = self.environment.action_space.shape[0]  # 17

        # initialize actor and critic
        self.actor = ActorCritic(input_dimensions=self.observation_dimensions, output_dimensions=self.action_dimensions)
        self.critic = ActorCritic(input_dimensions=self.observation_dimensions, output_dimensions=1)

        # initialize Adam optimizers for updating actor and critic parameters with backpropagation
        self.actor_optimizer = AdamOptimizer(params=self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = AdamOptimizer(params=self.critic.parameters(), lr=self.learning_rate)

        # initialize covariance matrix for representing Diagonal Gaussian Policy
        self.covariance_matrix = torch.diag(input=torch.full(size=(self.action_dimensions,), fill_value=0.5),
                                            diagonal=0)

    def __init__hyper_parameters(self, hyper_parameters):
        """
        Initialize hyper-parameter values.
        https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml

        :param hyper_parameters: set of algorithm parameters
        :return: None, just initializes parameters as variables which can be used in ProximalPolicyOptimization class
        """
        self.learning_rate = 0.005  # learning rate for actor's Adam optimizer
        self.number_of_time_steps_per_batch = 4800  # number of time-steps that will be run per one batch
        self.maximum_number_of_time_steps_per_episode = 1600
        self.number_of_network_updates_per_iteration = 5  # number of times to update actor/critic network per iteration

        # Discounting factor gamma which is used for calculating Reward-to-Go Policy Gradient
        self.discounting_factor = 0.95

        # rendering during training will be set to False, during testing to True in order to see the results
        self.render = False

        self.seed = None  # sets seed for ProximalPolicyOptimization class which can lead to different results

        if self.seed is not None:
            assert (type(self.seed) == int)

            torch.manual_seed(self.seed)

        # normalize advantage function (decrease variance, faster convergence and more stable model)
        self.normalize = True

        self.clip = 0.2  # generally recommended in the literature

    def get_action(self, observation):
        """
        The two most common kinds of stochastic policies in deep RL are categorical policies and diagonal Gaussian
        policies. Categorical policies can be used in discrete action spaces, while diagonal Gaussian policies are used
        in continuous action spaces.
        Two key computations are centrally important for using and training stochastic policies:
            * sampling actions from the policy and
            * computing log likelihoods of particular actions, log(PI_theta (a|s) )

        A multivariate Gaussian distribution (multivariate normal distribution) is described by a mean vector, mu, and a
        covariance matrix, sigma. A diagonal Gaussian distribution is a special case where the covariance matrix only
        has entries on the diagonal. As a result, we can represent it by a vector.
        A diagonal Gaussian policy always has a neural network that maps from observations to mean actions, mu_theta(s).
        There are two different ways that the covariance matrix is typically represented.


        :param observation: current state
        :return: an action in that state and logarithmic probability for that action in distribution
        """

        # query the actor network for a "mean" action (forward propagation)
        mu = self.actor.forward(observation)

        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.covariance_matrix)

        # sample an action from the distribution close to our mean
        action = multivariate_gaussian_distribution.sample()

        # calculate logarithmic probability for that action
        logarithmic_probability = multivariate_gaussian_distribution.log_prob(value=action)

        return action.detach().numpy(), logarithmic_probability.detach()

    def rewards_to_go(self, batch_rewards):
        """
        Computes reward to go for each time-step in a batch given the rewards.
        Agents should really only reinforce actions on the basis of their consequences. Rewards obtained before taking
        an action have no bearing on how good that action was: only rewards that come after. Actions are only reinforced
        based on rewards obtained after they are taken. We’ll call this form the “reward-to-go policy gradient,”
        because the sum of rewards after a point in a trajectory is called the reward-to-go from that point, and this
        policy gradient expression depends on the reward-to-go from state-action pairs.

        :param batch_rewards: rewards in one batch, shape = (NUMBER_OF_EPISODES, NUMBER_OF_TIME_STEPS_PER_EPISODE)
        :return: rewards to go, shape = (NUMBER_OF_TIME_STEPS)
        """
        rewards_to_go = []

        for episode_rewards in reversed(batch_rewards):
            # iterate through all rewards in the episode
            discounted_sum_of_rewards = 0

            for reward in reversed(episode_rewards):
                discounted_sum_of_rewards = reward + discounted_sum_of_rewards * self.discounting_factor
                rewards_to_go.insert(0, discounted_sum_of_rewards)

        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32)

        return rewards_to_go

    def collect_trajectories(self):
        """
        Collects set of trajectories D_k = {tau_i} by running policy PI_k = PI(theta_k) in the environment
        One rollout represents one trajectory and set of trajectories represents one batch.

        :return:
        """
        batch_data = BatchData()

        t = 0  # number of time-steps

        while t < self.number_of_time_steps_per_batch:
            rewards_per_episode = []

            # Reset the environment
            observation = self.environment.reset()
            is_done = False
            episode_t = None  # t-th time-step during an episode

            # Run an episode for a maximum number of time steps per episode
            for episode_t in range(self.maximum_number_of_time_steps_per_episode):
                if self.render:
                    self.environment.render()

                t += 1

                batch_data.observations.append(observation)

                action, logarithmic_probability = self.get_action(observation=observation)

                observation, reward, is_done, info = self.environment.step(action)

                # save reward, action and action's logarithmic probability
                rewards_per_episode.append(reward)
                batch_data.actions.append(action)
                batch_data.logarithmic_probabilities.append(logarithmic_probability)

                # if episode has finished before a given maximum number of time steps
                if is_done:
                    break

            batch_data.lengths_of_episodes.append(episode_t + 1)
            batch_data.rewards.append(rewards_per_episode)

        batch_observations = torch.tensor(batch_data.observations, dtype=torch.float32)
        batch_actions = torch.tensor(batch_data.actions, dtype=torch.float32)
        batch_logarithmic_probabilities = torch.tensor(batch_data.logarithmic_probabilities, dtype=torch.float32)
        batch_rewards_to_go = self.rewards_to_go(batch_rewards=batch_data.rewards)

        return \
            batch_observations, batch_actions, batch_logarithmic_probabilities, \
            batch_rewards_to_go, batch_data.lengths_of_episodes

    def calculate_value_function(self, batch_observations, batch_actions):
        """
        Estimate the value function for each observation in a batch of observations and logarithmic probabilities of the
        actions taken/executed in that batch with the most recent iteration of the actor network.

        :param batch_observations: shape = (NUMBER_OF_TIME_STEPS, OBSERVATION_DIMENSION)
        :param batch_actions: shape = (NUMBER_OF_TIME_STEPS, ACTION_DIMENSION)
        :return: function values for every observation and logarithmic probabilities for every action in recent iter
        """
        V = self.critic.forward(batch_observations)

        V = V.squeeze()  # (TIME_STEPS_PER_BATCH, 1) => (TIME_STEPS_PER_BATCH)

        mu = self.actor.forward(batch_observations)

        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.covariance_matrix)

        logarithmic_probability = multivariate_gaussian_distribution.log_prob(value=batch_actions)

        return V, logarithmic_probability

    def normalize_advantage_function(self, advantage_function, epsilon):
        """
        Standardization of advantage function with epsilon addition.

        :param advantage_function: advantage function for normalization
        :param epsilon: small number
        :return: normalized advantage function
        """
        return (advantage_function - advantage_function.mean()) / (advantage_function.std() + epsilon)

    def calculate_pi_theta_ratio(self, pi_theta, pi_theta_k):
        return torch.exp(pi_theta - pi_theta_k)

    def train(self, K):
        """

        :param K: total number of time-steps
        :return:
        """
        # for k = 0, 1, 2, ... K do
        k = 0
        i = 0  # iterations

        while k < K:
            batch_observations, batch_actions, batch_logarithmic_probabilities, batch_rewards_to_go, \
            lengths_of_episodes = self.collect_trajectories()

            k += np.sum(lengths_of_episodes)

            i += 1

            V, _ = self.calculate_value_function(batch_observations, batch_actions)
            advantage_function_k = batch_rewards_to_go - V.detach()

            if self.normalize:
                advantage_function_k = self.normalize_advantage_function(advantage_function=advantage_function_k,
                                                                         epsilon=1e-10)

            # update networks for some number of epochs
            for _ in range(self.number_of_network_updates_per_iteration):
                """
                Actor network (theta parameters):
                    Update the policy by maximizing the PPO-clip objective via stochastic gradient ascent with Adam.
                Critic network (phi parameters):
                    Fit value function by regression on mean-squared error via gradient descent with Adam.
                """
                V, pi_theta = self.calculate_value_function(batch_observations=batch_observations,
                                                            batch_actions=batch_actions)

                pi_ratios = self.calculate_pi_theta_ratio(pi_theta, batch_logarithmic_probabilities)

                first_term = pi_ratios * advantage_function_k
                second_term = torch.clamp(pi_ratios, 1 - self.clip, 1 + self.clip) * advantage_function_k

                # Adam minimizes the loss (thus the negative sign), take mean to get single loss as a float
                ppo_clip_objective = (-torch.min(first_term, second_term)).mean()  # maximization
                self.actor_optimizer.zero_grad()
                ppo_clip_objective.backward(retain_graph=True)  # https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
                self.actor_optimizer.step()

                regression_mean_squared_error = nn.MSELoss()(V, batch_rewards_to_go)
                self.critic_optimizer.zero_grad()
                regression_mean_squared_error.backward()
                self.critic_optimizer.step()


if __name__ == "__main__":
    import gym
    import pybulletgym

    env = gym.make('HumanoidPyBulletEnv-v0')
    model = ProximalPolicyOptimization(env)
    model.train(10000)
