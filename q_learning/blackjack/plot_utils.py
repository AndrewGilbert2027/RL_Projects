import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, algorithm):
    """
    Plot the rewards over episodes.
    
    :param rewards: List of total rewards per episode.
    :param algorithm: Name of the algorithm used for training.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label=f'Rewards ({algorithm})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Total Rewards per Episode - {algorithm}')
    plt.legend()
    plt.grid()
    plt.show()

def plot_average_rewards(rewards, window_size=100, algorithm='Algorithm'):
    """
    Plot the average rewards over a sliding window.
    
    :param rewards: List of total rewards per episode.
    :param window_size: Size of the sliding window for averaging.
    :param algorithm: Name of the algorithm used for training.
    """
    if len(rewards) < window_size:
        print("Not enough data to compute average rewards.")
        return
    
    average_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(window_size - 1, len(rewards)), average_rewards, label=f'Average Rewards ({algorithm})')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.title(f'Average Total Rewards per Episode - {algorithm}')
    plt.legend()
    plt.grid()
    plt.savefig(f'./results/average_rewards_{algorithm}.png')
