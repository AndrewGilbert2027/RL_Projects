import numpy as np
import gymnasium as gym

def create_q_table(state_size, action_size):
    """
    Creates a Q-table initialized with zeros.

    Args:
        state_size (int): The number of possible states.
        action_size (int): The number of possible actions.

    Returns:
        np.ndarray: A 2D array representing the Q-table.
    """
    return np.zeros((32*11*2, action_size.n))