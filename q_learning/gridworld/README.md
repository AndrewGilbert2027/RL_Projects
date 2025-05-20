# Gridworld Test Environment

The **Gridworld Test Environment** is a simple reinforcement learning environment implemented in C++. It is designed to demonstrate Q-learning in a grid-based world where an agent navigates from a starting position to a goal position while maximizing rewards.

---

## 📂 File Structure

- **`grid.h`**: Defines the `Grid` class, which represents the gridworld environment, and related enums (`Actions`, `Rewards`) and structs (`Information`).
- **`grid.c++`**: Implements the methods of the `Grid` class, including movement logic, grid resetting, and reward calculation.
- **`train_util.h`**: Contains utility functions for Q-learning, such as initializing the Q-table, selecting actions using epsilon-greedy strategy, and updating the Q-values.
- **`train.c++`**: The main entry point for training the agent using Q-learning. It initializes the environment, trains the agent over multiple episodes, and logs the results.
- **`Makefile`**: A build script to compile the project using `g++`.
- **`README.md`**: This file, explaining the environment and its components.

---

## 🗺️ Environment Description

The gridworld is a 2D grid where the agent starts at a specified position and must navigate to a goal position. The environment includes the following features:

- **Grid Dimensions**: The grid is defined by the number of rows and columns.
- **Start Position**: The initial position of the agent.
- **Goal Position**: The target position the agent must reach to complete an episode.
- **Actions**: The agent can take one of the following actions:
  - `UP`
  - `DOWN`
  - `LEFT`
  - `RIGHT`
  - `STAY` (no movement)
- **Rewards**:
  - `GOAL_REWARD` (+10): Awarded when the agent reaches the goal.
  - `STEP_PENALTY` (-1): Penalized for each step taken.
  - `INVALID_MOVE_PENALTY` (-5): Penalized for attempting to move outside the grid boundaries.

---

## 🚀 How It Works

1. **Grid Initialization**:
   - The grid is initialized with the specified dimensions, start position, and goal position.
   - The goal position is marked with a `G`, and the agent's position is displayed as `A` during visualization.

2. **Agent Movement**:
   - The agent moves based on the selected action.
   - If the move is valid, the agent's position is updated. If invalid, a penalty is applied.

3. **Q-Learning**:
   - The agent learns an optimal policy using the Q-learning algorithm.
   - The Q-table is updated using the formula:
     ```
     Q(s, a) = Q(s, a) + α * (reward + γ * max_a' Q(s', a') - Q(s, a))
     ```
   - Exploration is controlled using an epsilon-greedy strategy.

4. **Termination**:
   - An episode ends when the agent reaches the goal or exceeds the maximum number of steps.

---

## 🛠️ Building and Running

1. **Build the Project**:
   Run the following command in the `gridworld` directory:
   ```bash
   make