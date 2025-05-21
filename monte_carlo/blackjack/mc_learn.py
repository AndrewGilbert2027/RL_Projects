import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import BlackjackAgent

# Initialize the environment
env = gym.make("Blackjack-v1", natural=True) # natural=True for natural blackjack

# Set the number of episodes
num_episodes = 100000

# Initialize the number of states and actions
state_space = env.observation_space
action_space = env.action_space


# Initialize the agent
epsilon = 1.0 # Exploration rate
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
        action = agent.get_action_epsilon_greedy(state)
        
        # Take the action and observe the next state and reward
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Store the state, action, and reward in the episode data
        episode_data.append((state, action, reward))
        
        # Move to the next state
        state = next_state

    # Update the agent with the episode data
    agent.update(episode_data)

    # Decay epsilon
    if episode % 1000 == 0:
        agent.epsilon = max(0.01, agent.epsilon * 0.9)  # Decay epsilon



# Create a rolling average of the rewards
window_size = 1000
rolling_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
# Plot the rolling average of the rewards
plt.plot(rolling_avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Rolling Average of Rewards')
plt.savefig('rolling_avg_rewards.png')