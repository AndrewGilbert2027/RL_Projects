import argparse
import numpy as np
import gymnasium as gym
import rl_utils as ru
from agent import Agent
from blackJackEnv import BlackJackEnv

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run a Q-learning algorithm for Lunar Lander.")
parser.add_argument("--algorithm", type=str, required=True, help="Algorithm to use: basic_q_learning, double_q_learning, sarsa, dqn, or td(n)")
parser.add_argument("--n", type=int, required=False, help="Number of steps for TD(n) (only required if algorithm is td(n))")
args = parser.parse_args()

algorithm = args.algorithm
if algorithm not in ["basic_q_learning", "double_q_learning", "sarsa", "dqn", "td(n)"]:
    raise ValueError("Invalid algorithm specified. Choose from: basic_q_learning, double_q_learning, sarsa, dqn, or td(n).")

if algorithm == "td(n)" and args.n is None:
    raise ValueError("The --n argument is required when using the td(n) algorithm.")

n = args.n if algorithm == "td(n)" else None

print(f"Using algorithm: {algorithm}")
if n:
    print(f"Number of steps for TD(n): {n}")

# Create the Blackjack environment
env = BlackJackEnv(
    num_decks=1,
    natural_blackjack=True,
)

# Initialize the agent
agent = Agent(env, algorithm=algorithm, alpha=0.1, gamma=1.0)

# Run the agent
agent.run(
    num_episodes=1000000,
    max_steps=100,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
)


# Plot the rewards
import plot_utils as pu
pu.plot_average_rewards(agent.rewards, window_size=10000, algorithm=algorithm)

