from algorithm import ProximalPolicyOptimization
import gym
import pybulletgym
from os.path import exists


def train_model(environment, hyper_parameters, total_number_of_time_steps, save_frequency, save_model_path,
                actor_model, critic_model, logging_path):
    """
    Function for training the model for a specified gym environment.

    :param environment:  gym environment of interest
    :param hyper_parameters: hyper-parameters for model training
    :param total_number_of_time_steps: total number of time-steps
    :param save_frequency: how frequent to save the model
    :param save_model_path: file path where we will save the model
    :param actor_model: actor model file path (if non-existing, then new model is trained)
    :param critic_model: critic model file path (if non-existing, then new model is trained)
    :param logging_path: where to log training information
    :return: None, saves model at specified location
    """
    print("Training the model...\n")

    if not exists(path=actor_model) and not exists(path=critic_model):
        print("Training model for the first time (actor and critic models don't exist).")

    model = ProximalPolicyOptimization(environment=environment, save_frequency=save_frequency,
                                       save_model_path=save_model_path, logging_path=logging_path,
                                       actor_path=actor_model, critic_path=critic_model, **hyper_parameters)

    model.train(K=total_number_of_time_steps)


if __name__ == "__main__":
    env = gym.make('HumanoidPyBulletEnv-v0')

    model_hyper_parameters = {
        "learning_rate": 2.5e-4,
        "number_of_time_steps_per_batch": 4800,
        "maximum_number_of_time_steps_per_episode": 1600,
        "number_of_network_updates_per_iteration": 20,
        "discounting_factor": 0.99,
        "render": False,
        "seed": None,
        "normalize": True,
        "clip_range": 0.2
    }

    train_model(environment=env, hyper_parameters=model_hyper_parameters, total_number_of_time_steps=1e12,
                save_frequency=5, save_model_path="../trained_models/ppo",
                actor_model="../trained_models/ppo/ppo_actor.pth", critic_model="../trained_models/ppo/ppo_critic.pth",
                logging_path="../plots")
