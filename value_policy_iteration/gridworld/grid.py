import numpy as np
from enum import Enum

# Default 4x4 map
default_map = [
    ['s', '.', '.', '.'],
    ['h', '.', '.', 'h'],
    ['.', '.', '.', '.'],
    ['.', 'h', '.', 'g']
]

# Define actions as an enumeration for better readability
class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

# Define Actions as tupe showing how to move on grid
class Action_Change(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)


# Define transitions of where we end up moving to
# Even though same as Actions we want to maintain readability
class Transitions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

# Define rewards as an enumeration for better readability
class Rewards(Enum):
    GOAL = 10
    HOLE = -10
    EMPTY = 0
    WALL = -5



class GridWorld:
    def __init__(self, map=None, is_slippery=False):
        """
        Initialize the GridWorld environment.
        :param map: 2D list representing the grid. 's' for start, 'g' for goal, 'h' for holes, '.' for empty spaces.
        :param is_slippery: Boolean indicating if the environment is slippery (stochastic) (default: False).
        """
        if map is None:
            map = default_map
        self.map = map
        self.is_slippery = is_slippery
        self.grid = np.array(self.map)
        self.rows = self.grid.shape[0]
        self.cols = self.grid.shape[1]
        self.start = self.get_start()
        self.goal = self.get_goal()
        self.holes = self.get_holes()
        self.action_space = 5
        self.state_space = np.sum(self.grid.shape) # get how many possible states

    def get_transition_probabilities(self, state, action):
        transition_probabilities = np.zeros(self.state_space) # array representing the action.
        if not self.is_slippery:
            next_state = self.move(state, action)
            next_state = next_state[1] * self.cols + next_state[0] # current_row * cols + current_col
            transition_probabilities[next_state] = 1.0
        else:
            transition_probabilities = self.stochastic_action(state, action)

        return transition_probabilities

    def move(self, state, action):
        """
        Move the agent deterministically based on the given action.
        :param state: Tuple (row, column) representing the current state.
        :param action: Action to be taken (from the Actions enum).
        :return: Tuple (row, column) representing the next state.
        """
        dx, dy = Action_Change[action.name].value  # Get the change in position for the action
        new_state = (state[0] + dx, state[1] + dy)

        # Check if the new state is valid (within bounds and not a hole)
        if self.__is_valid(new_state):
            return new_state
        else:
            return state  # Return the original state if the move is invalid
        
    def stochastic_action(self, state, action):
        """
        Compute the stochastic transition probabilities for a given state and action.
        :param state: Tuple (row, column) representing the current state.
        :param action: Action to be taken (from the Actions enum).
        :return: A (1, state_space) array representing the transition probabilities.
        """
        transition_probabilities = np.zeros(self.state_space)  # Initialize probabilities to zero

        # Define the primary action and its left/right alternatives
        primary_action = action
        left_action = Actions((action.value - 1) % len(Actions))  # Action to the left
        right_action = Actions((action.value + 1) % len(Actions))  # Action to the right

        # Compute the resulting states for each action
        primary_state = self.move(state, primary_action)
        left_state = self.move(state, left_action)
        right_state = self.move(state, right_action)

        # Convert states to indices in the state space
        primary_index = primary_state[1] * self.cols + primary_state[0]
        left_index = left_state[1] * self.cols + left_state[0]
        right_index = right_state[1] * self.cols + right_state[0]

        # Assign probabilities
        transition_probabilities[primary_index] += 0.8  # 80% chance of taking the primary action
        transition_probabilities[left_index] += 0.1  # 10% chance of taking the left action
        transition_probabilities[right_index] += 0.1  # 10% chance of taking the right action

        # Handle out-of-bounds or invalid moves
        total_probability = np.sum(transition_probabilities)
        if total_probability < 1.0:
            # Distribute the remaining probability across valid states
            remaining_probability = 1.0 - total_probability
            valid_states = np.argwhere(transition_probabilities > 0).flatten()
            for index in valid_states:
                transition_probabilities[index] += remaining_probability / len(valid_states)

        return transition_probabilities

    def get_start(self):
        """
        Get the starting position in the grid.
        :return: Tuple (row, column) of the starting position.
        :raises ValueError: If start position 's' is not found in the grid.
        """
        start = np.argwhere(self.grid == 's')
        if start.size == 0:
            raise ValueError("Start position 's' not found in the grid.")
        return tuple(start[0])
    
    def get_goal(self):
        """
        Get the goal position in the grid.
        :return: Tuple (row, column) of the goal position.
        :raises ValueError: If goal position 'g' is not found in the grid.
        """
        goal = np.argwhere(self.grid == 'g')
        if goal.size == 0:
            raise ValueError("Goal position 'g' not found in the grid.")
        return tuple(goal[0])
    
    def get_holes(self):
        """
        Get the positions of holes in the grid.
        :return: List of tuples representing the positions of holes.
        :raises ValueError: If holes are invalid (not reachable from start).
        """
        holes = np.argwhere(self.grid == 'h')
        if holes.size == 0:
            return []
        
        if not self.__valid_holes():
            raise ValueError("Invalid holes in the grid. (Could not reach goal from start)")
        return [tuple(hole) for hole in holes]
    
    def __valid_holes(self):
        """
        Check if the holes are valid (reachable from start).
        :return: True if holes are valid, False otherwise.
        """
        # Perform DFS to check if goal is reachable from start
        start = self.start
        goal = self.goal
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == goal:
                return True
            visited.add(current)
            for neighbor in self.__get_neighbors(current):
                if neighbor not in visited and self.grid[neighbor] != 'h':
                    stack.append(neighbor)
        return False
    
    def __get_neighbors(self, pos):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self.__is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    

    def __is_valid(self, pos):
        return (0 <= pos[1] < self.grid.shape[0] and
                0 <= pos[0] < self.grid.shape[1])