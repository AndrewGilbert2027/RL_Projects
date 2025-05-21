import numpy as np
from enum import Enum

default_map = np.array(['s...', 'h..h', 'h...', 'h.h.', '...g'], dtype='<U4')

class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    STAY = (0, 0)

class GridWorld:
    def __init__(self, map=None, is_slippery=False):
        self.map = map if map is not None else default_map
        self.grid = np.array([list(row) for row in self.map])
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 5  # up, down, left, right, stay
        self.__valid_map()
        self.is_slippery = is_slippery

    def policy_iteration(self, discount_factor=0.9, theta=1e-6):
        """
        Policy Iteration algorithm to find the optimal policy and value function.
        :param discount_factor: Discount factor for future rewards.
        :param theta: Threshold for convergence.
        :return: Tuple (optimal policy, optimal value function).
        """

        policy = np.zeros((self.n_rows, self.n_cols), dtype=int)  # Initialize policy (Up for all states)
        value_function = np.zeros((self.n_rows, self.n_cols))  # Initialize value function
        while True:
            new_value_function, new_policy = self.__get_new_value_function(value_function, discount_factor)
            delta = np.max(np.abs(new_value_function - value_function))
            value_function = new_value_function
            policy = new_policy
            if delta < theta:
                break

        return policy, value_function
    
    def __get_new_value_function(self, value_function, discount_factor):
        """
        Calculate the new value function and policy based on the current value function.
        :param value_function: Current value function.
        :param discount_factor: Discount factor for future rewards.
        :return: Tuple (new value function, new policy).
        """
        new_value_function = np.zeros_like(value_function)
        new_policy = np.zeros_like(value_function, dtype=int)

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                state = (row, col)
                action_values = np.zeros(self.n_actions)
                for action in range(self.n_actions):
                    action_values[action] = self.__calculate_value(state, action, value_function, discount_factor)
                new_value_function[state] = np.max(action_values)
                new_policy[state] = np.argmax(action_values) # index of the action with the highest value

        return new_value_function, new_policy

    def __calculate_value(self, state, action, value_function, discount_factor):
        """
        Calculate the value of a state-action pair.
        :param state: Tuple (row, column) representing the current state.
        :param action: Integer representing the action to take.
        :param value_function: Current value function.
        :param discount_factor: Discount factor for future rewards.
        :return: Value of the state-action pair.
        """
        if state[0] < 0 or state[0] >= self.n_rows or state[1] < 0 or state[1] >= self.n_cols:
            raise ValueError("State out of bounds.")
        if action < 0 or action >= self.n_actions:
            raise ValueError("Invalid action.")

        transition_probabilities = self.get_transition_probabilities(state, action)
        transition_probabilities = transition_probabilities.flatten()
        value = 0
        for next_state_index, prob in enumerate(transition_probabilities):
            if prob > 0:
                next_state = next_state_index // self.n_cols, next_state_index % self.n_cols
                value += prob * (self.__reward(next_state) + discount_factor * value_function[next_state])
        return value
    
    def __reward(self, state):
        """
        Get the reward for a given state.
        :param state: Tuple (row, column) representing the current state.
        :return: Reward for the state.
        """
        if self.grid[state] == 'g':
            return 10.0
        elif self.grid[state] == 'h':
            return -8.0
        else:
            return -1.0  # Default reward for empty space
        


    def get_transition_probabilities(self, state, action):
        """
        Get the transition probabilities for a given state and action.
        :param state: Tuple (row, column) representing the current state.
        :param action: Integer representing the action to take.
        :return: List of tuples (next_state, probability).
        """
        if state[0] < 0 or state[0] >= self.n_rows or state[1] < 0 or state[1] >= self.n_cols:
            raise ValueError("State out of bounds.")
        if action < 0 or action >= self.n_actions:
            raise ValueError("Invalid action.")
        
        transition_probabilities = np.zeros((self.n_rows, self.n_cols))
        if action == 0:
            action_enum = Action.UP
        elif action == 1:
            action_enum = Action.DOWN
        elif action == 2:
            action_enum = Action.LEFT
        elif action == 3:
            action_enum = Action.RIGHT
        elif action == 4:
            action_enum = Action.STAY
        action_vector = action_enum.value
        if not self.is_slippery:
            next_state = (state[0] + action_vector[0], state[1] + action_vector[1])
            if self.__is_valid(next_state):
                transition_probabilities[next_state] = 1.0 # valid action (did not hit wall)
            else:
                transition_probabilities[state] = 1.0 # hit a wall
            return transition_probabilities

        else:
            # Slippery case: 80% chance to go in the intended direction
            # If intended direction was out of bounds, then spread probability to either left or right relative to move
            # 10% chance to go left or right
            intended_direction = action_vector
            if (intended_direction[0] == 0 and intended_direction[1] == 0):
                # Stay action
                transition_probabilities[state] = 1.0
                return transition_probabilities
            elif (intended_direction[0] != 0): 
                left_direction = (0, -1)
                right_direction = (0, 1)
                left_state = (state[0] + left_direction[0], state[1] + left_direction[1])
                right_state = (state[0] + right_direction[0], state[1] + right_direction[1])
                if self.__is_valid(left_state):
                    transition_probabilities[left_state] += 0.1
                if self.__is_valid(right_state):
                    transition_probabilities[state] += 0.1
            else:
                left_direction = (-1, 0)
                right_direction = (1, 0)
                left_state = (state[0] + left_direction[0], state[1] + left_direction[1])
                right_state = (state[0] + right_direction[0], state[1] + right_direction[1])
                if self.__is_valid(left_state):
                    transition_probabilities[left_state] += 0.1
                if self.__is_valid(right_state):
                    transition_probabilities[state] += 0.1
            
            next_state = (state[0] + intended_direction[0], state[1] + intended_direction[1])
            if self.__is_valid(next_state):
                transition_probabilities[next_state] += 0.8
            # Normalize the probabilities (Helps with case where left and right actinos are invalid)
            return transition_probabilities / np.sum(transition_probabilities)

    def __valid_map(self):
        """
        Validate the map to ensure it meets the required conditions.
        """
        # Check if the map is a numpy array
        if not isinstance(self.map, np.ndarray):
            raise ValueError("Map must be a numpy array.")

        # Check if the map is a 2D array
        if self.grid.ndim != 2:
            raise ValueError("Map must be a 2D array.")

        # Check if the map is at least 2x2
        if self.grid.shape[0] < 2 or self.grid.shape[1] < 2:
            raise ValueError("Map must be at least 2x2.")

        # Check if all elements in the grid are valid characters
        valid_characters = {'s', 'g', 'h', '.'}
        if not np.all(np.isin(self.grid, list(valid_characters))):
            raise ValueError("Map must contain only 's', 'g', '.', and 'h' characters.")

        # Check if there is exactly one start state ('s')
        if np.count_nonzero(self.grid == 's') != 1:
            raise ValueError("Map must contain exactly one start state 's'.")

        # Check if there is exactly one goal state ('g')
        if np.count_nonzero(self.grid == 'g') != 1:
            raise ValueError("Map must contain exactly one goal state 'g'.")

        # Check for invalid holes (holes that make the goal unreachable)
        if self.__invalid_holes():
            raise ValueError("Map contains invalid holes (cannot reach goal state).")

    def __invalid_holes(self):
        # Check if there are holes that cannot be reached
        # Mark all reachable states from the start state
        visited = np.zeros_like(self.grid, dtype=bool)
        start_pos = np.argwhere(self.grid == 's')[0]
        visited[start_pos[0], start_pos[1]] = True
        stack = [start_pos]
        while stack:
            current_pos = stack.pop()
            if self.grid[current_pos[0], current_pos[1]] == 'g':
                return False
            for neighbor in self.__get_neighbors(current_pos):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return True

    def __get_neighbors(self, pos):
        """
        Get all valid neighboring positions for a given position.
        :param pos: Tuple (row, column) representing the current position.
        :return: List of tuples representing valid neighboring positions.
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        for dx, dy in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.__is_valid(new_pos):
                neighbors.append(new_pos)

        return neighbors

    def __is_valid(self, pos):
        """
        Check if a position is valid (within bounds and not a hole).
        :param pos: Tuple (row, column) representing the position to check.
        :return: True if the position is valid, False otherwise.
        """
        return (0 <= pos[0] < self.n_rows and
                0 <= pos[1] < self.n_cols)


