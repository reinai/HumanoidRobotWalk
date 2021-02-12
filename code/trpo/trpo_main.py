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
    def __init__(self, env, actor, critic, delta, gamma, cg_delta, alpha, backtrack_steps_num, gae_lambda, critic_epoch_num):
        """
        Initialize the parameters of TRPO class
        Args:
        :param env: environment which we will solve using trpo
        :param actor: actor model
        :param critic: critic model
        :param delta: delta used as kl divergence constraint
        :param gamma: discount factor
        :param cg_delta: conjugate gradient offset
        :param alpha: factor to compute max step to update actor parameters
        :param backtrack_steps_num: number of steps to compute max step to update actor parameters
        :param gae_lambda: gae factor
        :param critic_epoch_num: number of epoch to train critic nn
        """
        self.actor = actor
        self.critic = critic
        self.delta = delta
        self.env = env
        self.gamma = gamma
        self.cg_delta = cg_delta
        self.alpha = alpha
        self.backtrack_steps_num = backtrack_steps_num
        self.gae_lambda = gae_lambda
        self.critic_epoch_num = critic_epoch_num

    def estimate_advantages(self, states, rewards):
        """ Estimating advantage based on rewards and value of states obtained through critic"""
        values = self.critic.model.forward(states)
        returns = torch.zeros_like(rewards)
        gae = 0
        for i in reversed(range(rewards.shape[0])):
            if i == rewards.shape[0] - 1:
                delta = rewards[i] - values[i]
            else:
                delta = rewards[i] + self.gamma * values[i+1] - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns[i] = gae + values[i]
        advantages = returns - values
        return advantages

    def surrogate_loss(self, new_probs, old_probs, advantages):
        """ Defining surrogate loss"""
        return (torch.exp(new_probs - old_probs) * advantages).mean()

    def kl_divergence(self, p, q):
        """ Defining KL divergence """
        return torch.square((p - q)).sum().mean()

    def compute_grad(self, y, x, retain_graph=False, create_graph=False):
        """ Computing gradient dy/dx"""
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    def conjugate_gradient(self, hvp_function, b, max_iterations = 10):
        """ Conjugate gradient algorithm to compute H^-1*b"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        i = 0
        while i < max_iterations:
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
        """ Get advantage estimation that is in right form and normalized """
        advantages = [self.estimate_advantages(states, rewards) for states, _, rewards, _, _ in episodes]
        advantages = torch.cat(advantages, dim=0).flatten()
        return (advantages - advantages.mean()) / advantages.std()

    def get_probability(self, actions, states):
        mu = self.actor.model.forward(states)
        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.actor.covariance_matrix)
        logarithmic_probability = multivariate_gaussian_distribution.log_prob(value=actions)
        return logarithmic_probability

    def update_agent(self, episodes):
        """ Method to update agent that we train """
        #PART 1: get states and actions provided through parameter episodes
        states = torch.cat([r.states for r in episodes], dim=0)
        next_states = torch.cat([r.next_states for r in episodes], dim=0)
        actions = torch.cat([r.actions for r in episodes], dim=0)

        #PART 2: calculate advantages based on trajectories and normalize it
        advantages = self.get_advantage_estimation(episodes)
        #PART 3: update critic parameters based on advantage estimation
        for iter in range(self.critic_epoch_num):
            train_advantages = self.get_advantage_estimation(episodes)
            self.critic.update_critic(train_advantages)
        #PART 4: get distribution of the policy and define surrogate loss and kl divergence


        probability = self.get_probability(actions, states)
        distribution = self.actor.model.forward(states)


        L = self.surrogate_loss(probability, probability.detach(), advantages)
        KL = self.kl_divergence(distribution, distribution.detach())
        #PART 5: compute gradient for surrogate loss and kl divergence
        parameters = list(self.actor.model.parameters())
        g = self.compute_grad(L, parameters, retain_graph=True)
        d_kl = self.compute_grad(KL, parameters, create_graph=True)
        #PART 6: define hessian vector product, compute search direction and max_length to get max step
        def HVP(v):
            return self.compute_grad(d_kl @ v, parameters, retain_graph=True)
        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.delta / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir
        #PART 7: check if max step satisfy the constraint, if not make it smaller
        def criterion(step):
            self.actor.upgrade_parameters(step)
            with torch.no_grad():
                distribution_new = self.actor.model.forward(states)
                probability_new = self.get_probability(actions, states)
                L_new = self.surrogate_loss(probability_new, probability, advantages)
                KL_new = self.kl_divergence(distribution, distribution_new)
            L_improvement = L_new - L
            if L_improvement > 0 and KL_new <= self.delta:
                return True
            self.actor.upgrade_parameters(-step)
            return False
        i = 0
        while not criterion((self.alpha ** i) * max_step) and i < self.backtrack_steps_num:
            i += 1

    def train(self, epochs, num_of_episodes, render_frequency = None, max_reward_per_episode = 5000):
        """ Training an agent """
        mean_total_rewards = []
        global_episode = 0
        for epoch in range(epochs):
            episodes = []
            episode_total_rewards = []
            for t in range(num_of_episodes):
                state = self.env.reset()
                done = False
                samples = []
                episode_reward = 0
                while not done and episode_reward < max_reward_per_episode:
                    if render_frequency is not None and global_episode % render_frequency == 0:
                        self.env.render()
                    action, probability = self.actor.get_action(state)
                    curr_action = action.numpy()
                    next_state, reward, done, _ = self.env.step(curr_action)
                    episode_reward += reward
                    samples.append((state, action, reward, next_state, probability))
                    state = next_state
                states, actions, rewards, next_states, probabilities = zip(*samples)
                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
                actions = torch.stack([torch.tensor(action) for action in actions], dim=0).float()
                rewards = torch.as_tensor(rewards).unsqueeze(1)
                probabilities = torch.tensor(probabilities, requires_grad = True).unsqueeze(1)
                episodes.append(Episode(states, actions, rewards, next_states, probabilities))
                episode_total_rewards.append(rewards.sum().item())
                global_episode += 1
            self.update_agent(episodes)
            mtr = np.mean(episode_total_rewards)
            print(f'E: {epoch}.\tMean total reward across {num_of_episodes} episodes: {mtr}')
            mean_total_rewards.append(mtr)
            if epoch % 10 == 9:
                torch.save(self.actor.model.state_dict(), 'HumanoidRobotWalk/code/trpo/models/actor' + str(epoch + 1) + '.pt')
                torch.save(self.critic.model.state_dict(), 'HumanoidRobotWalk/code/trpo/models/critic' + str(epoch + 1) + '.pt')
                with open('HumanoidRobotWalk/code/trpo/models/rewards' + str(epoch + 1) + '.txt', 'w+') as fp:
                    fp.write(str(mean_total_rewards))
        plt.plot(mean_total_rewards)
        plt.show()


if __name__ == "__main__":
    env = gym.make('HumanoidPyBulletEnv-v0')
    #env.render()
    env.reset()
    actor = Actor(44, 17)
    #actor.model.load_state_dict(torch.load('HumanoidRobotWalk/code/trpo/actor490.pt'))
    critic = Critic(44, 1)
    #critic.model.load_state_dict(torch.load('HumanoidRobotWalk/code/trpo/critic490.pt'))
    trpo = TRPO(env=env,
                actor=actor,
                critic=critic,
                delta=0.003,
                gamma=0.995,
                cg_delta=0.1,
                alpha=0.9,
                backtrack_steps_num=10,
                gae_lambda=0.98,
                critic_epoch_num = 10)
    trpo.train(epochs=10,num_of_episodes=10,render_frequency=None)
