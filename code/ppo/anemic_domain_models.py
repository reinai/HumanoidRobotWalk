"""
Anemic domain model is the use of a software domain model where the domain objects contain no business logic
(validations, calculations, business rules etc.).
"""


class BatchData(object):
    def __init__(self):
        # all observations/states that were collected/experienced in this batch
        self.observations = []  # shape = (NUMBER_OF_TIME_STEPS, OBSERVATION_DIMENSION)

        # all actions that were collected/taken in this batch
        self.actions = []  # shape = (NUMBER_OF_TIME_STEPS, ACTION_DIMENSION)

        # logarithmic probability of each collected/taken action in this batch
        self.logarithmic_probabilities = []  # shape = (NUMBER_OF_TIME_STEPS)

        # all rewards collected during every time-step in one batch
        self.rewards = []  # shape = (NUMBER_OF_TIME_STEPS)

        # all rewards-to-go for each time-step in one batch
        self.rewards_to_go = []  # shape = (NUMBER_OF_TIME_STEPS)

        # lengths of episodes in one batch
        self.lengths_of_episodes = []  # shape = (NUMBER_OF_EPISODES)
