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
        self.start = self.get_start()
        self.goal = self.get_goal()
        self.holes = self.get_holes()

    def get_transition_probabilities(self, state, action):
        return None # Placeholder for transition probabilities

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
        return (0 <= pos[0] < self.grid.shape[0] and
                0 <= pos[1] < self.grid.shape[1] and
                self.grid[pos] != 'h')