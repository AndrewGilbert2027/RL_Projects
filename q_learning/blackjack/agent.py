import numpy as np
import gymnasium as gym
import rl_utils as ru


class Agent:
    def __init__(self, env: gym.Env, alpha: float = 0.1, gamma: float = 0.99, algorithm= "basic_q_learning"):
        """
        Initialize the agent with the environment, learning rate, and discount factor.
        :param env: The environment in which the agent operates.
        :param alpha: Learning rate for the agent.
        :param gamma: Discount factor for future rewards.
        :param discretize_fn: Function to discretize continuous states.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.action_space = env.action_space
        self.state_space = env.observation_space
        self.q_table = np.zeros((32, 11, 2, env.action_space))  # (player_sum, dealer_card, usable_ace, actions)
        self.rewards = [] # Store total discounted rewards for each episode (for plotting or analysis)
        self.algorithm = algorithm
        if algorithm == "double_q_learning":
            self.q_table2 = np.zeros_like(self.q_table)  # Second Q-table for Double Q-learning


    def run(self, num_episodes: int, max_steps: int, epsilon_start: float, epsilon_end: float, epsilon_decay: float):
        """
        Run the agent in the environment for a specified number of episodes.
        :param
        num_episodes: Number of episodes to run the agent.
        :param max_steps: Maximum number of steps per episode.
        :param epsilon_start: Initial value of epsilon for exploration.
        :param epsilon_end: Minimum value of epsilon for exploration.
        :param epsilon_decay: Decay rate for epsilon.
        """
        if self.algorithm == "basic_q_learning":
            self.train_basic_q(num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay)
        elif self.algorithm == "double_q_learning":
            self.train_double_q(num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay)
        elif self.algorithm == "sarsa":
            self.train_sarsa(num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay)
        elif self.algorithm == "dqn":
            raise NotImplementedError("Deep Q-Network (DQN) is not implemented yet.")
        else:
            raise ValueError("Invalid algorithm specified. Choose from: basic_q_learning, double_q_learning, sarsa, or dqn.")


    def choose_action_greedy(self, state: tuple) -> int:
        """
        Choose an action using a greedy policy based on the Q-table.
        :param state: The current state of the environment.
        :return: The action to take.
        """
        return np.argmax(self.q_table[state])
    
    def choose_action_epsilon_greedy(self, state: tuple, epsilon: float) -> int:
        """
        Choose an action using an epsilon-greedy
        policy based on the Q-table.
        :param state: The current state of the environment.
        :param epsilon: Probability of choosing a random action.
        :return: The action to take.
        """
        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])  # Randomly choose between 'hit' (0) and 'stand' (1)
        else:
            return self.choose_action_greedy(state)
        
    def choose_action_greedy_double_q(self, state: tuple) -> int:
        return np.argmax(self.q_table[state] + self.q_table2[state])
    
    def choose_action_epsilon_greedy_double_q(self, state: tuple, epsilon: float) -> int:
        """
        Choose an action using an epsilon-greedy
        policy based on the Double Q-learning Q-tables.
        :param state: The current state of the environment.
        :param epsilon: Probability of choosing a random action.
        :return: The action to take.
        """
        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])
        else:
            return self.choose_action_greedy_double_q(state)
        
    def update_q_table_double_q(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Update the Q-table using the Double Q-learning
        update rule.
        :param state: The current state of the environment.
        :param action: The action taken.
        :param reward: The reward received after taking the action.
        :param next_state: The next state of the environment.
        """
        if np.random.rand() < 0.5:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table2[next_state][best_next_action]
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error
        else:
            best_next_action = np.argmax(self.q_table2[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
            td_error = td_target - self.q_table2[state][action]
            self.q_table2[state][action] += self.alpha * td_error
        

    def update_q_table_basic(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Update the Q-table using the Q-learning
        update rule.
        :param state: The current state of the environment.
        :param action: The action taken.
        :param reward: The reward received after taking the action.
        :param next_state: The next state of the environment.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def update_q_table_sarsa(self, state: tuple, action: int, reward: float, next_state: tuple, next_action: int):
        """
        Update the Q-table using the SARSA update rule.
        :param state: The current state of the environment.
        :param action: The action taken.
        :param reward: The reward received after taking the action.
        :param next_state: The next state of the environment.
        :param next_action: The action taken in the next state.
        """
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def reset(self):
        """
        Reset the agent's Q-table.
        This is useful for starting a new episode or training session.
        """
        self.q_table = ru.create_q_table(self.env.observation_space.n, self.action_space.n)

    def decay_epsilon(self, epsilon: float, decay_rate: float) -> float:
        """
        Decay the epsilon value for epsilon-greedy action selection.
        :param epsilon: The current epsilon value.
        :param decay_rate: The rate at which to decay epsilon.
        :return: The decayed epsilon value.
        """
        self.epsilon =  max(0.01, epsilon * decay_rate)

    def save_q_table(self, filename: str):
        """
        Save the Q-table to a file.
        :param
        filename: The name of the file to save the Q-table to.
        """
        np.save(filename, self.q_table)

    def load_q_table(self, filename: str):
        """
        Load the Q-table from a file.
        :param filename: The name of the file to load the Q-table from.
        """
        self.q_table = np.load(filename)
        if self.q_table.shape != (self.env.observation_space.n, self.action_space.n):
            raise ValueError("Loaded Q-table has incorrect shape.")
        
    def train_sarsa(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay):
        """
        Train the agent using the SARSA algorithm.
        :param num_episodes: Number of episodes to train the agent.
        :param max_steps: Maximum number of steps per episode.
        :param epsilon_start: Initial value of epsilon for exploration.
        :param epsilon_end: Minimum value of epsilon for exploration.
        :param epsilon_decay: Decay rate for epsilon.
        """
        if (epsilon_start <= 0 or epsilon_end <= 0 or epsilon_decay <= 0 or 
            epsilon_start <= epsilon_end):
            raise ValueError("Epsilon values must be positive and start > end.")
        
        if (num_episodes <= 0 or max_steps <= 0):
            raise ValueError("Number of episodes and maximum steps must be positive integers.")

        epsilon = epsilon_start
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            player_sum, dealer_card, usable_ace = state

            # Clamp state values to valid ranges
            player_sum = max(0, min(31, player_sum))
            dealer_card = max(1, min(10, dealer_card))

            total_reward = 0
            
            action = self.choose_action_epsilon_greedy((player_sum, dealer_card - 1, int(usable_ace)), epsilon)
            
            for step in range(max_steps):
                next_state, reward, done, _, _ = self.env.step(action)
                next_player_sum, next_dealer_card, next_usable_ace = next_state

                # Clamp next state values to valid ranges
                next_player_sum = max(0, min(31, next_player_sum))
                next_dealer_card = max(1, min(10, next_dealer_card))

                # Adjust next_dealer_card to 0-based index for Q-table
                next_state_index = (next_player_sum, next_dealer_card - 1, int(next_usable_ace))
                
                next_action = self.choose_action_epsilon_greedy(next_state_index, epsilon)
                
                self.update_q_table_sarsa((player_sum, dealer_card - 1, int(usable_ace)), action, reward, next_state_index, next_action)
                
                state = next_state
                action = next_action
                total_reward += reward
                
                if done:
                    break
            
            # Store the total discounted reward for each episode
            self.rewards.append

            if (episode + 1) % 10000 == 0:
                # Decay epsilon after each episode
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
                
            self.rewards.append(total_reward)

    def train_double_q(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay):
        if (epsilon_start <= 0 or epsilon_end <= 0 or epsilon_decay <= 0 or 
            epsilon_start <= epsilon_end):
            raise ValueError("Epsilon values must be positive and start > end.")
        if (num_episodes <= 0 or max_steps <= 0):
            raise ValueError("Number of episodes and maximum steps must be positive integers.")
        if (self.algorithm != "double_q_learning"):
            raise ValueError("Algorithm must be set to 'double_q_learning' for this method.")
        epsilon = epsilon_start

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            player_sum, dealer_card, usable_ace = state

            # Clamp state values to valid ranges
            player_sum = max(0, min(31, player_sum))
            dealer_card = max(1, min(10, dealer_card))

            total_reward = 0
            
            action = self.choose_action_epsilon_greedy_double_q((player_sum, dealer_card - 1, int(usable_ace)), epsilon)
            
            for step in range(max_steps):
                next_state, reward, done, _, _ = self.env.step(action)
                next_player_sum, next_dealer_card, next_usable_ace = next_state

                # Clamp next state values to valid ranges
                next_player_sum = max(0, min(31, next_player_sum))
                next_dealer_card = max(1, min(10, next_dealer_card))

                # Adjust next_dealer_card to 0-based index for Q-table
                next_state_index = (next_player_sum, next_dealer_card - 1, int(next_usable_ace))
                
                next_action = self.choose_action_epsilon_greedy_double_q(next_state_index, epsilon)
                
                self.update_q_table_double_q((player_sum, dealer_card - 1, int(usable_ace)), action, reward, next_state_index)
                
                state = next_state
                action = next_action
                total_reward += reward
                
                if done:
                    break
            
            # Store the total discounted reward for each episode
            self.rewards.append(total_reward)

            if (episode + 1) % 10000 == 0:
                # Decay epsilon after each episode
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        
        

    def train_basic_q(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay):
        """
        Train the agent using the basic Q-learning algorithm.
        :param num_episodes: Number of episodes to train the agent.
        :param max_steps: Maximum number of steps per episode.
        :param epsilon_start: Initial value of epsilon for exploration.
        :param epsilon_end: Minimum value of epsilon for exploration.
        :param epsilon_decay: Decay rate for epsilon.
        """
        if (epsilon_start <= 0 or epsilon_end <= 0 or epsilon_decay <= 0 or 
            epsilon_start <= epsilon_end):
            raise ValueError("Epsilon values must be positive and start > end.")
        
        if (num_episodes <= 0 or max_steps <= 0):
            raise ValueError("Number of episodes and maximum steps must be positive integers.")

        epsilon = epsilon_start
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            player_sum, dealer_card, usable_ace = state

            # Clamp state values to valid ranges
            player_sum = max(0, min(31, player_sum))
            dealer_card = max(1, min(10, dealer_card))

            total_reward = 0
            
            for step in range(max_steps):
                # Adjust dealer_card to 0-based index for Q-table
                state_index = (player_sum, dealer_card - 1, int(usable_ace))
                action = self.choose_action_epsilon_greedy(state_index, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                next_player_sum, next_dealer_card, next_usable_ace = next_state

                # Clamp next state values to valid ranges
                next_player_sum = max(0, min(31, next_player_sum))
                next_dealer_card = max(1, min(10, next_dealer_card))

                # Adjust next_dealer_card to 0-based index for Q-table
                next_state_index = (next_player_sum, next_dealer_card - 1, int(next_usable_ace))
                self.update_q_table_basic(state_index, action, reward, next_state_index)
                
                state = next_state
                total_reward = total_reward * self.gamma + reward
                
                if done:
                    break
            

            # Store the total discounted reward for each episode
            self.rewards.append(total_reward)
            
            if (episode + 1) % 10000 == 0:
                # Decay epsilon after each episode
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")



