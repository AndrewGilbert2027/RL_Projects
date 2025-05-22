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

        # Initialize the Q-table as a dictionary
        # Keys are state-action pairs, values are Q-values
        self.q_table = {}

        # Initialize a counter for state-action pairs
        # Keys are state-action pairs, values are counts
        self.counter = {}

    @property
    def q_values(self):
        """
        Convert the Q-table into a dictionary format compatible with create_grids.
        :return: A dictionary where keys are states and values are arrays of Q-values for each action.
        """
        q_values = {}
        for (state, action), value in self.q_table.items():
            if state not in q_values:
                q_values[state] = np.zeros(self.action_space.n)
            q_values[state][action] = value
        return q_values

    def get_action_epsilon_greedy(self, state):
        """
        Choose an action using the epsilon-greedy policy.
        :param state: The current state.
        :return: The chosen action.
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = [self.q_table.get((state, action), 0) for action in range(self.action_space.n)]
            return np.argmax(q_values)

    def update(self, episode_data):
        """
        Update the Q-table using Monte Carlo On-Policy Control.
        :param episode_data: A list of (state, action, reward) tuples from the episode.
        """
        g = 0  # Initialize the return
        for state, action, reward in reversed(episode_data):
            g = reward + self.gamma * g  # Update the return
            if (state, action) not in [(s, a) for s, a, _ in episode_data[:-1]]:
                # Increment the counter for the state-action pair
                self.counter[(state, action)] = self.counter.get((state, action), 0) + 1

                # Update the Q-value incrementally
                if (state, action) not in self.q_table:
                    self.q_table[(state, action)] = 0
                self.q_table[(state, action)] += (g - self.q_table[(state, action)]) / self.counter[(state, action)]

    def decay(self):
        """
        Decay the exploration rate epsilon.
        """
        self.epsilon = max(0.1, self.epsilon * 0.99)