import gym
import pybulletgym
import time

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
