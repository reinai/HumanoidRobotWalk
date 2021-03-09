# HumanoidRobotWalk
Implementation of Trust Region Policy Optimization and Proximal Policy Optimization algorithms on the objective of Robot Walk.

## Programs & libraries needed in order to run this project 
* [OpenAI Gym](https://gym.openai.com/) : A toolkit for developing and comparing reinforcement learning algorithms
* [PyBullet Gym](https://github.com/benelot/pybullet-gym) : PyBullet Robotics Environments fully compatible with Gym toolkit (uses the Bullet physics engine)
* [PyTorch](https://pytorch.org/) : Open source machine learning library based on the Torch library
* [NumPy](https://www.numpy.org/) : Fundamental package for scientific computing with Python
* [matplotlib](https://matplotlib.org/) : Plotting library for the Python programming language and its numerical mathematics extension NumPy

## Algorithms pseudocodes
### Trust Region Policy Optimization (TRPO) - implemented by [Vasilije Pantić](https://github.com/sovaso)
![alt text](https://raw.githubusercontent.com/reinai/HumanoidRobotWalk/main/utils/trpo.png)
### Proximal Policy Optimization (PPO) - implemented by [Nikola Zubić](https://github.com/nikolazubic)
![alt text](https://raw.githubusercontent.com/reinai/HumanoidRobotWalk/main/utils/ppo.png)

## How to run?
For TRPO: Run `trpo_main.py` at `root/code/trpo/`,<br>
For PPO: Run `ppo_main.py` at `root/code/ppo/`,<br>
and enter the absolute file path to the trained model.<br><br>
Trained models are available at: `root/code/trained_models/`.

# In motion
TO DO

## Numerical results
TO DO

Copyright (c) 2021 Nikola Zubić, Vasilije Pantić
