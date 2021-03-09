import gym
import pybulletgym
import torch
import time
from actor import Actor
from critic import Critic
import sys

# models/actor40900.pth
actor_model_path = input("Enter actor model file path: ")

env = gym.make('HumanoidPyBulletEnv-v0')
env.render()
observation = env.reset()

observation_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.shape[0]

actor_neural_network = Actor(input_dim=observation_dimensions, output_dim=action_dimensions)
actor_neural_network.model.load_state_dict(torch.load(actor_model_path))

while True:
    action = actor_neural_network.model.forward(observation).detach().numpy()
    observation, reward, is_done, info = env.step(action)
    time.sleep(0.03)

    if is_done is True:
        env.reset()

    env.render()
