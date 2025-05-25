import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from collections import defaultdict
from agent import BlackjackAgent
from plot_utils import plot_policy_and_values

# Initialize the environment
env = gym.make("Blackjack-v1", natural=True) # natural=True for natural blackjack

# Set the number of episodes
num_episodes = 1000000

# Initialize the number of states and actions
state_space = 32 * 11 * 2  # Flattened state space (Player sum, Dealer showing, Usable ace)
action_space = 2  # Number of actions (Hit, Stick)

# Initialize the agent
epsilon = 0.9 # Exploration rate
gamma = 1.0 # Discount factor (For Blackjack, we can set it to 1)
agent = BlackjackAgent(state_space=state_space, action_space=action_space, epsilon=epsilon, gamma=gamma)

rewards = []

# Implement the Monte Carlo On-Policy Control algorithm
for episode in range(num_episodes):
    # Reset the environment
    state, info = env.reset()
    done = False
    episode_data = []  # Store the episode data (state, action, reward)

    # Generate an episode
    while not done:
        # Choose an action using epsilon-greedy policy
        state_index = state[0] * 11 * 2 + state[1] * 2 + int(state[2])  # Convert state to a flattened index
        action = agent.get_action_epsilon_greedy(state_index)
        
        # Take the action and observe the next state and reward
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        next_state_index = next_state[0] * 11 * 2 + next_state[1] * 2 + int(next_state[2])
        
        # Store the state, action, and reward in the episode data
        episode_data.append((state_index, action, reward))
        
        # Move to the next state
        state = next_state

    # Update the agent with the episode data
    agent.update(episode_data)

    # Decay epsilon
    if episode % 10000 == 0:
        agent.decay()
        print(f"Episode {episode}: Epsilon decayed to {agent.epsilon}")

# Create a rolling average of the rewards
window_size = 1000
rolling_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
# Plot the rolling average of the rewards
plt.plot(rolling_avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Rolling Average of Rewards')
plt.savefig('rolling_avg_rewards.png')


"""
Code below was taken from https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py
"""


def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # Convert state-action values to state values and build a policy dictionary
    state_value = defaultdict(float)
    policy = defaultdict(int)

    for state, action_values in enumerate(agent.q_values):
        # Convert the flattened state index back to (player_sum, dealer_showing, usable_ace)
        player_sum = state // (11 * 2)
        dealer_showing = (state % (11 * 2)) // 2
        ace_flag = state % 2

        if ace_flag == usable_ace:
            state_value[(player_sum, dealer_showing, usable_ace)] = float(np.max(action_values))
            policy[(player_sum, dealer_showing, usable_ace)] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),  # Player's count
        np.arange(1, 11),   # Dealer's face-up card
    )

    # Create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value.get((obs[0], obs[1], usable_ace), 0),
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # Create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy.get((obs[0], obs[1], usable_ace), 0),
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # Create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # Plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # Plot the policy
    ax2 = fig.add_subplot(1, 2, 2)
    sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False, ax=ax2)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # Add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

# Visualize state values & policy with usable ace
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With Usable Ace")
plt.savefig("policy_with_usable_ace.png")

# Visualize state values & policy without usable ace
value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without Usable Ace")
plt.savefig("policy_without_usable_ace.png")