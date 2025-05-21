import matplotlib.pyplot as plt
import numpy as np

def visualize_gridworld(gridworld, policy=None, value_function=None):
    """
    Visualize the GridWorld environment, policy, and value function.
    :param gridworld: The GridWorld object.
    :param policy: A 2D array representing the policy (optional).
    :param value_function: A 2D array representing the value function (optional).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, gridworld.n_cols)
    ax.set_ylim(0, gridworld.n_rows)
    ax.set_xticks(np.arange(0, gridworld.n_cols + 1, 1))
    ax.set_yticks(np.arange(0, gridworld.n_rows + 1, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.invert_yaxis()  # Invert y-axis to match grid indexing

    # Draw the grid
    for row in range(gridworld.n_rows):
        for col in range(gridworld.n_cols):
            state = gridworld.grid[row, col]
            if state == 's':
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='green', alpha=0.5))  # Start
                ax.text(col + 0.5, row + 0.5, 'S', ha='center', va='center', fontsize=12, color='black')
            elif state == 'g':
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='gold', alpha=0.5))  # Goal
                ax.text(col + 0.5, row + 0.5, 'G', ha='center', va='center', fontsize=12, color='black')
            elif state == 'h':
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='red', alpha=0.5))  # Hole
                ax.text(col + 0.5, row + 0.5, 'H', ha='center', va='center', fontsize=12, color='black')
            else:
                ax.add_patch(plt.Rectangle((col, row), 1, 1, color='white', alpha=0.5))  # Empty space

            # Add value function if provided
            if value_function is not None:
                ax.text(col + 0.5, row + 0.8, f"{value_function[row, col]:.2f}", ha='center', va='center', fontsize=8, color='blue')

            # Add policy arrows if provided
            if policy is not None:
                action = policy[row, col]
                if action == 0:  # UP
                    ax.arrow(col + 0.5, row + 0.5, 0, -0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 1:  # DOWN
                    ax.arrow(col + 0.5, row + 0.5, 0, 0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 2:  # LEFT
                    ax.arrow(col + 0.5, row + 0.5, -0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 3:  # RIGHT
                    ax.arrow(col + 0.5, row + 0.5, 0.3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 4:  # STAY
                    ax.text(col + 0.5, row + 0.5, '•', ha='center', va='center', fontsize=12, color='black')

    plt.title("GridWorld Visualization")
    plt.savefig("gridworld_visualization.png")