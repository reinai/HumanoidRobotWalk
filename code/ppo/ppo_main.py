# Execution of trained PPO model on "HumanoidPyBulletEnv-v0"
import gym
import pybulletgym
import torch
import time
from ppo.actor_critic import ActorCritic
import sys

# /home/nikola/Desktop/agt_projekat/HumanoidRobotWalk/code/trained_models/ppo/48 hours/ppo_actor.pth

actor_model_path = input("Enter actor model file path: ")

env = gym.make('HumanoidPyBulletEnv-v0')
env.render()
observation = env.reset()

observation_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.shape[0]

actor_neural_network = ActorCritic(input_dimensions=observation_dimensions, output_dimensions=action_dimensions)

actor_neural_network.load_state_dict(torch.load(actor_model_path))

while True:
    action = actor_neural_network.forward(observation).detach().numpy()
    observation, reward, is_done, info = env.step(action)
    time.sleep(0.03)

    if is_done is True:
        env.reset()

    env.render()
