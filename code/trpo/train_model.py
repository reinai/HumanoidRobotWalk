import gym
import pybulletgym
import time
import torch
from actor import Actor
from critic import Critic
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.distributions import MultivariateNormal

Episode = namedtuple('Episode', ['states', 'actions', 'rewards', 'next_states', 'probabilities'])


class TRPO():
    """ Main class that implements trpo algorithm to improve actor and critic neural networks """

    def __init__(self, env, actor, critic, delta, gamma, cg_delta, cg_iterations, alpha, backtrack_steps_num, critic_epoch_num):
        """
        Initialize the parameters of TRPO class
        Args:
        :param env: environment which we will solve using trpo algorithm
        :param actor: actor model for this problem that is used as a policy function
        :param critic: critic model for this problem that is used as a value state function
        :param delta: number used as a KL divergence constraint between two distributions
        :param gamma: discount factor
        :param cg_delta: conjugate gradient constraint to tell us when to stop with the process
        :param cg_iterations: maximum number of iterations for conjugate gradient algortihm
        :param alpha: factor to compute max step to update actor parameters in order to satisfy the KL divergence constraint (delta)
        :param backtrack_steps_num: number of steps to compute max step to update actor parameters
        :param critic_epoch_num: number of epoch to train critic neural network
        """

        self.actor = actor
        self.critic = critic
        self.delta = delta
        self.env = env
        self.gamma = gamma
        self.cg_delta = cg_delta
        self.cg_iterations = cg_iterations
        self.alpha = alpha
        self.backtrack_steps_num = backtrack_steps_num
        self.critic_epoch_num = critic_epoch_num

    def estimate_advantages(self, states, rewards):
        """
        Estimating the advantage based on trajectories for one episode

        :param states: states we visited during the episode
        :param rewards: collected rewards in that concrete episode
        :return: estimated advantage
        """

        # using critic nn to get state values
        values = self.critic.model.forward(states)
        # defining a variable to store rewards-to-go values
        rtg = torch.zeros_like(rewards)
        # setting last value on zero
        last_value = 0
        # calculating rewards-to-go
        for i in reversed(range(rewards.shape[0])):
            last_value = rtg[i] = rewards[i] + self.gamma * last_value
        # advantage = rewards-to-go - values
        return rtg - values

    def surrogate_loss(self, new_probs, old_probs, advantages):
        """
        Calculating surrogate loss that is used for calculating policy network gradients.
        Formula: mean(e^(p1-p2)*adv), where p1 and p2 are log probabilities

        :param new_probs: log probabilities of the new policy
        :param old_probs: log probabilities of the old policy
        :param advantages: estimated advantage of all episodes of current epoch
        :return: surrogate loss
        """
        return (torch.exp(new_probs - old_probs) * advantages).mean()

    def kl_divergence(self, mean1, std1, mean2, std2):
        """
        Calculating the KL divergence between two distributions (old and new policy)
        Formula: log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5

        :param mean1: new means
        :param std1: new standard deviations
        :param mean2: old means
        :param std2: old standard deviations
        :return: KL divergence between two distributions
        """

        # log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5

        mean2 = mean2.detach()
        std2 = std2.detach()
        kl_div = torch.log(std1) - torch.log(std2) + (std2.pow(2) + (mean1 - mean2).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl_div.sum(1, keepdim=True).mean()

    def compute_grad(self, y, x, retain_graph=False, create_graph=False):
        """
        Calculating the derivative of y with respect to x -> dx/dy

        :param y: function
        :param x: parameter
        :param retain_graph: boolean value should we retain a graph
        :param create_graph: boolean value to define should we create a graph
        :return: derivation dy/dx
        """

        if create_graph:
            retain_graph = True
        grad = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        grad = torch.cat([t.view(-1) for t in grad])
        return grad

    def conjugate_gradient(self, hvp_function, b):
        """
        Calculate the H^1 * g using the conjugate gradient algorithm

        :param hvp_function: hessian vector product function
        :param b: vector that will be multiplied with inverse hessian matrix
        :return: multiplication of vector g and inverse hessian matrix H^-1
        """

        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        i = 0
        while i < self.cg_iterations:
            AVP = hvp_function(p)
            dot_old = r @ r
            alpha = dot_old / (p @ AVP)
            x = x + alpha * p
            r = r - alpha * AVP
            if r.norm() <= self.cg_delta:
                return x
            beta = (r @ r) / dot_old
            p = r + beta * p
            i += 1
        return x

    def get_advantage_estimation(self, episodes):
        """
        Function to gather all estimated advantages of each episode of the epoch in one variable and normalize it

        :param episodes: episodes of the epoch
        :return: estimated normalized advantages
        """

        #collect all advantages
        advantages = [self.estimate_advantages(states, rewards) for states, _, rewards, _, _ in episodes]
        advantages = torch.cat(advantages, dim=0).flatten()
        #normalizing the advantages
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def get_probability(self, actions, states):
        """
        Calculating logaritmic probability of actions based on states

        :param actions: actions of the trajectories
        :param states: states of the trajectories
        :return: logarithmic probability
        """

        #mean of the distribution
        mu = self.actor.model.forward(states)
        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.actor.covariance_matrix)
        logarithmic_probability = multivariate_gaussian_distribution.log_prob(value=actions)
        return logarithmic_probability

    def update_agent(self, episodes):
        """
        Function that update both critic and actor neural network's parameters

        :param episodes: episodes in the epoch
        """

        # PART 1: get states and actions provided through parameter episodes
        states = torch.cat([r.states for r in episodes], dim=0)
        actions = torch.cat([r.actions for r in episodes], dim=0)

        # PART 2: calculate advantages based on trajectories and normalize it
        advantages = self.get_advantage_estimation(episodes).detach()

        # PART 3: update critic parameters based on advantage estimation
        for iter in range(self.critic_epoch_num):
            self.critic.update_critic(self.get_advantage_estimation(episodes))

        # PART 4: get distribution of the policy and define surrogate loss and kl divergence
        probability = self.get_probability(actions, states)
        mean, std = self.actor.get_mean_std(states)
        L = self.surrogate_loss(probability, probability.detach(), advantages)
        KL = self.kl_divergence(mean, std, mean, std)

        # PART 5: compute gradient for surrogate loss and kl divergence
        parameters = list(self.actor.model.parameters())
        g = self.compute_grad(L, parameters, retain_graph=True)
        d_kl = self.compute_grad(KL, parameters, create_graph=True)
        #print('Gradient -> ', g)

        # PART 6: define hessian vector product function, compute search direction and max_length to get max step
        def HVP(v):
            return self.compute_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.delta / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        # PART 7: check if max step satisfy the constraint, if not make it smaller
        def criterion(step):
            #print('Step ->', step)
            self.actor.update_parameters(step)
            with torch.no_grad():
                mean_new, std_new = self.actor.get_mean_std(states)
                probability_new = self.get_probability(actions, states)
                L_new = self.surrogate_loss(probability_new, probability, advantages)
                KL_new = self.kl_divergence(mean_new, std_new, mean, std)
            L_improvement = L_new - L
            #print('Distribution difference ->', KL_new)
            #print('Loss improvement ->', L_new)
            if L_improvement > 0 and KL_new <= self.delta:
                return True
            self.actor.update_parameters(-step)
            return False

        i = 0
        while not criterion((self.alpha ** i) * max_step) and i < self.backtrack_steps_num:
            i += 1

    def train(self, epochs, num_of_timesteps, max_timesteps_per_episode, render_frequency=None, starting_with=0):
        """
        Function for running the trajectories and training neural networks using them

        :param epochs: number of epochs
        :param num_of_timesteps: number of timesteps in one epoch (because it is continuous world)
        :param max_timesteps_per_episode: maximal number of timestep for one episode
        :param render_frequency: render frequency in miliseconds
        :param starting_with: define a number of epoch to start with
        """

        mean_total_rewards = []
        global_episode = 0
        for epoch in range(epochs):

            episodes, episode_total_rewards = [], []
            curr_number_of_timesteps, num_of_episodes = 0, 0
            while curr_number_of_timesteps < num_of_timesteps:

                num_of_steps = 0
                state = self.env.reset()
                done = False
                samples = []
                curr_episode_steps = 0
                while not done and curr_episode_steps < max_timesteps_per_episode:

                    #rendering the environment
                    if render_frequency is not None and global_episode % render_frequency == 0:
                        self.env.render()
                    #getting action and probability of it based on current state
                    action, probability = self.actor.get_action(state)
                    curr_action = action.numpy()
                    #running current action in the environment
                    next_state, reward, done, _ = self.env.step(curr_action)
                    num_of_steps += 1
                    samples.append((state, action, reward, next_state, probability))
                    state = next_state
                    curr_episode_steps += 1

                num_of_episodes += 1
                curr_number_of_timesteps += curr_episode_steps
                states, actions, rewards, next_states, probabilities = zip(*samples)
                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
                actions = torch.stack([torch.tensor(action) for action in actions], dim=0).float()
                rewards = torch.as_tensor(rewards).unsqueeze(1)
                probabilities = torch.tensor(probabilities, requires_grad=True).unsqueeze(1)
                episodes.append(Episode(states, actions, rewards, next_states, probabilities))
                episode_total_rewards.append(rewards.sum().item())
                global_episode += 1

            #updating the agent
            self.update_agent(episodes)
            mtr = np.mean(episode_total_rewards)
            mean_total_rewards.append(mtr)
            #printing the statistics of current epoch
            print(f'E: {epoch+1+starting_with}.\tMean total reward across {num_of_episodes} episodes and {curr_number_of_timesteps} timesteps: {mtr}')

            #every 50 epoch, save all mean rewards and model for actor & critic
            if epoch % 50 == 49:
                torch.save(self.actor.model.state_dict(),
                           'HumanoidRobotWalk/code/trpo/models/actor' + str(epoch + starting_with + 1) + '.pt')
                torch.save(self.critic.model.state_dict(),
                           'HumanoidRobotWalk/code/trpo/models/critic' + str(epoch + starting_with + 1) + '.pt')
                with open('HumanoidRobotWalk/code/trpo/models/rewards' + str(epoch + starting_with + 1) + '.txt', 'w+') as fp:
                    fp.write(str(mean_total_rewards))

        #plotting the results
        plt.plot(mean_total_rewards)
        plt.show()


if __name__ == "__main__":
    # main funtion for training
    env = gym.make('HumanoidPyBulletEnv-v0')
    env.render()
    env.reset()
    actor = Actor(44, 17)
    actor.model.load_state_dict(torch.load('./models/actor16500.pt'))
    critic = Critic(44, 1, 2.5e-4)
    critic.model.load_state_dict(torch.load('./models/critic16500.pt'))
    trpo = TRPO(env=env,
                actor=actor,
                critic=critic,
                delta=1e-2,
                gamma=0.99,
                cg_delta=1e-2,
                cg_iterations = 10,
                alpha=0.99,
                backtrack_steps_num=100,
                critic_epoch_num=20)
    trpo.train(epochs=10000,
               num_of_timesteps=12000,
               max_timesteps_per_episode=1600,
               render_frequency=100,
               starting_with=16500)
