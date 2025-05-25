import numpy as np

class BlackjackAgent:
    def __init__(self, state_space, action_space, epsilon=1.0, gamma=1.0):
        """
        Initialize the BlackjackAgent.
        :param state_space: The state space of the environment.
        :param action_space: The action space of the environment.
        :param epsilon: Exploration rate for epsilon-greedy policy.
        :param gamma: Discount factor for future rewards.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma

        # Initialize the Q-table as a numpy array
        self.q_table = np.zeros((state_space, action_space))

        # Initialize a counter for state-action pairs as a numpy array
        self.counter = np.zeros((state_space, action_space), dtype=int)

    @property
    def q_values(self):
        """
        Convert the Q-table into a dictionary-like format.
        :return: A numpy array of Q-values.
        """
        return self.q_table

    def get_action_epsilon_greedy(self, state):
        """
        Choose an action using the epsilon-greedy policy.
        :param state: The current state.
        :return: The chosen action.
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.action_space)
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.q_table[state])

    def update(self, episode_data):
        """
        Update the Q-table using Monte Carlo On-Policy Control.
        :param episode_data: A list of (state, action, reward) tuples from the episode.
        """
        g = 0  # Initialize the return
        for state, action, reward in reversed(episode_data):
            g = reward + self.gamma * g  # Update the return
            count = self.counter[state, action]  # Get the count for the state-action pair
            q_value = self.q_table[state, action]  # Get the current Q-value
            self.q_table[state, action] = (q_value * count + g) / (count + 1)  # Update the Q-value
            self.counter[state, action] += 1  # Increment the count for the state-action pair

    def decay(self):
        """
        Decay the exploration rate epsilon.
        """
        self.epsilon = max(0.05, self.epsilon * 0.99)