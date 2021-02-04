import gym
import pybulletgym
import time
import torch
from actor import Actor
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

Episode = namedtuple('Episode', ['states', 'actions', 'rewards', 'next_states', ])


class TRPO():
    def __init__(self, env):
        self.actor = Actor(0.01)
        self.env = env

    def train(self, epochs = 10, num_of_episodes = 10, render_frequency = None):
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

            self.actor.update_actor(episodes)
            mtr = np.mean(episode_total_rewards)
            print(f'E: {epoch}.\tMean total reward across {num_of_episodes} episodes: {mtr}')

            mean_total_rewards.append(mtr)

        plt.plot(mean_total_rewards)
        plt.show()


if __name__ == "__main__":
    env = gym.make('HumanoidPyBulletEnv-v0')
    env.render()
    env.reset()
    trpo = TRPO(env)
    trpo.train(epochs=100,num_of_episodes=25,render_frequency=100)

"""
env = gym.make('HumanoidPyBulletEnv-v0')
env.render()
obs = env.reset()

while True:
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, is_done, info = env.step(action)
    time.sleep(0.05)

    if is_done is True:
        env.reset()

    env.render()
"""