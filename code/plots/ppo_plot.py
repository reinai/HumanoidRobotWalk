import matplotlib.pyplot as plt


def read_log(file_path, plot_title):
    """
    Format:
        Iteration: 1
        Elapsed time-steps: 4801
        Average Episodic Length: 20.004
        Average Episodic Returns: -45.007
        Average actor loss: -0.023353729397058487

    :param file_path: path to log file
    :param plot_title: plot title
    :return: None, draws plots
    """
    time_steps = []
    episodic_length = []
    episodic_return = []
    actor_loss = []

    file = open(file_path, "r")
    content = file.readlines()

    count = 0

    for line in content:
        if line.strip() == "":
            continue

        count += 1

        tokens = line.split(":")
        value = float(tokens[1].strip())

        if count == 2:
            time_steps.append(value)
        elif count == 3:
            episodic_length.append(value)
        elif count == 4:
            episodic_return.append(value)
        elif count == 5:
            count = 0
            actor_loss.append(value)

    plt.title(plot_title)
    plt.xlabel("Time-steps")
    plt.ylabel("Average episodic return/length")
    plt.plot(time_steps, episodic_return, 'g', label='Average episodic return')
    plt.plot(time_steps, episodic_length, 'b', label='Average episodic length')
    plt.legend()
    plt.show()
    plt.title(plot_title)
    plt.xlabel("Time-steps")
    plt.ylabel("Average actor loss")
    plt.plot(time_steps, actor_loss, 'r', label='Average actor loss')
    plt.show()


read_log(file_path="ppo_log_files/log_6_5_hours.txt", plot_title="First 6.5 hours of training")
read_log(file_path="ppo_log_files/log_27_hours.txt", plot_title="19.5-27 hours of training")
read_log(file_path="ppo_log_files/log_48_hours.txt", plot_title="40-48 hours of training")
