import numpy as np


class ModelStatistics(object):
    def __init__(self):
        self.elapsed_time_steps = 0
        self.elapsed_iterations = 0
        self.batch_lengths_of_episodes = []
        self.batch_episodic_returns = []
        self.actor_losses = []

    def __str__(self):
        str = ""

        average_episode_lengths = round(np.mean(self.batch_lengths_of_episodes), 3)
        average_episodic_returns = round(np.mean([np.sum(episodic_rewards)
                                                  for episodic_rewards in self.batch_episodic_returns]), 3)
        average_actor_loss = np.mean([losses.float().mean() for losses in self.actor_losses])

        print("Iteration: {}".format(self.elapsed_iterations))
        print("Elapsed time-steps: {}".format(self.elapsed_time_steps))
        print("Average Episodic Length: {}".format(average_episode_lengths))
        print("Average Episodic Returns: {}".format(average_episodic_returns))
        print("Average actor loss: {}\n".format(average_actor_loss))

        str += "Iteration: {}\n".format(self.elapsed_iterations)
        str += "Elapsed time-steps: {}\n".format(self.elapsed_time_steps)
        str += "Average Episodic Length: {}\n".format(average_episode_lengths)
        str += "Average Episodic Returns: {}\n".format(average_episodic_returns)
        str += "Average actor loss: {}\n\n".format(average_actor_loss)

        self.batch_lengths_of_episodes = []
        self.batch_episodic_returns = []
        self.actor_losses = []

        return str
