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

        # Initialize the returns dictionary for Monte Carlo updates
        # Keys are state-action pairs, values are lists of returns
        self.returns = {}

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
        # Calculate the returns for each state-action pair in the episode
        g = 0  # Initialize the return
        for state, action, reward in reversed(episode_data):
            g = reward + self.gamma * g  # Update the return
            if (state, action) not in [(s, a) for s, a, _ in episode_data[:-1]]:
                # Only update if this is the first occurrence of the state-action pair in the episode
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(g)
                # Update the Q-value as the average of the returns
                self.q_table[(state, action)] = np.mean(self.returns[(state, action)])