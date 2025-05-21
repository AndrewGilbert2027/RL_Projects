# Blackjack Monte Carlo Simulation

## Overview
This project demonstrates the use of Monte Carlo reinforcement learning techniques to optimize strategies for the game of Blackjack. The simulation models the game environment and uses policy evaluation and improvement to enhance decision-making.

## Features
- **Environment**: The agent plays Blackjack in the `Blackjack-v1` environment provided by `gymnasium`. The environment includes features like natural blackjack and the ability to hit, stand, or bust.
- **Algorithm**: Monte Carlo On-Policy Control is used to estimate the optimal policy. The agent explores the environment using an epsilon-greedy policy and updates its Q-values based on the returns observed from complete episodes. Over time, the epsilon value decays until it reaches min value. Furthermore, because our take_action() function chooses based on the max q_value, it helps simplify the process of updating our policy because initializing it to the average of returns does this for us. 
- **Visualization**: The project visualizes the rolling average of rewards over episodes, showing how the agent's performance improves as it converges to the optimal policy.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/reinforcement-learning-projects.git
   cd reinforcement-learning-projects/RL_Projects/monte_carlo/blackjack
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running
To start the simulation, execute the main script:
```bash
python main.py
```



## Room for improvement
1. Compare different Monte Carlo Algorithms 
2. Improve visualization techniques (give a grid showcasing what action to take given what state)