from grid import GridWorld
from visual import visualize_gridworld
import numpy as np
# Create a GridWorld environment

# map = np.array(['s....g', 'h....h', 'h.h..h', '.h...h', 'h....h'], dtype='<U6')  

grid = GridWorld(is_slippery=True)

# Find the optimal policy using value iteration
policy, V = grid.policy_iteration(discount_factor=0.99, theta=1e-6)

print(policy)

# Visualize the GridWorld with the optimal policy and value function
visualize_gridworld(grid, policy=policy, value_function=V)