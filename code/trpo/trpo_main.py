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

Episode = namedtuple('Episode', ['states', 'actions', 'rewards', 'next_states', ])


class TRPO():
    def __init__(self, env, actor, critic, delta):
        self.actor = actor
        self.critic = critic
        self.delta = delta
        self.env = env

    def estimate_advantages(self, states, last_state, rewards):
        values = self.critic.model(states)
        last_value = self.critic.model(last_state.unsqueeze(0))
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        return advantages

    def surrogate_loss(self, new_probs, old_probs, advantages):
        return (new_probs / old_probs * advantages).mean()

    def kl_divergence(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()

    def calculate_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    def conjugate_gradient(self, hvp_function, b, cg_delta = 0.1, max_steps = 10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        i = 0
        while i < max_steps:
            AVP = hvp_function(p)
            dot_old = r @ r
            alpha = dot_old / (p @ AVP)
            x = x + alpha * p
            r = r - alpha * AVP
            if r.norm() <= cg_delta:
                return x
            beta = (r @ r) / dot_old
            p = r + beta * p
            i += 1
        return x

    def update_agent(self, episodes):
        states = torch.cat([r.states for r in episodes], dim=0)
        actions = torch.cat([r.actions for r in episodes], dim=0).flatten()
        advantages = [self.estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in
                      episodes]
        advantages = torch.cat(advantages, dim=0).flatten()
        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()
        self.critic.update_critic(advantages)
        distribution = self.actor.model.forward(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]
        L = self.surrogate_loss(probabilities, probabilities.detach(), advantages)
        KL = self.kl_divergence(distribution, distribution)
        parameters = list(self.actor.model.parameters())
        g = self.calculate_grad(L, parameters, retain_graph=True)
        d_kl = self.calculate_grad(KL, parameters, create_graph=True)

        def HVP(v):
            return self.calculate_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.delta / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            self.actor.upgrade_parameters(step)
            with torch.no_grad():
                distribution_new = self.actor.model.forward(states)
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]
                L_new = self.surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = self.kl_divergence(distribution, distribution_new)
            L_improvement = L_new - L
            if L_improvement > 0 and KL_new <= self.delta:
                return True
            self.actor.upgrade_parameters(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

    def train(self, epochs, num_of_episodes, render_frequency = None):
        mean_total_rewards = []
        global_episode = 0

        for epoch in range(epochs):
            episodes = []
            episode_total_rewards = []

            for t in range(num_of_episodes):
                state = self.env.reset()
                done = False

                samples = []

                while not done:
                    if render_frequency is not None and global_episode % render_frequency == 0:
                        self.env.render()

                    with torch.no_grad():
                        action = self.actor.get_action(state)
                        a = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        a[action - 1] = 1

                    next_state, reward, done, _ = self.env.step(a)

                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

                # Transpose our samples
                states, actions, rewards, next_states = zip(*samples)

                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
                actions = torch.as_tensor(actions).unsqueeze(1)
                rewards = torch.as_tensor(rewards).unsqueeze(1)

                episodes.append(Episode(states, actions, rewards, next_states))

                episode_total_rewards.append(rewards.sum().item())
                global_episode += 1

            self.update_agent(episodes)
            mtr = np.mean(episode_total_rewards)
            print(f'E: {epoch}.\tMean total reward across {num_of_episodes} episodes: {mtr}')

            mean_total_rewards.append(mtr)

            if epoch == epoch:
                torch.save(self.actor.model.state_dict(), './models/actor' + str(epoch) + '.pt')
                torch.save(self.critic.model.state_dict(), './models/critic' + str(epoch) + '.pt')
                with open('./models/rewards' + str(epoch) + '.txt', 'w+') as fp:
                    fp.write(str(mean_total_rewards))

        plt.plot(mean_total_rewards)
        plt.show()


if __name__ == "__main__":
    env = gym.make('HumanoidPyBulletEnv-v0')
    #env.render()
    env.reset()
    actor = Actor(44, 17)
    critic = Critic(44, 1)
    delta = 0.07
    trpo = TRPO(env=env, actor=actor, critic=critic, delta=delta)
    trpo.train(epochs=10,num_of_episodes=2048,render_frequency=100)