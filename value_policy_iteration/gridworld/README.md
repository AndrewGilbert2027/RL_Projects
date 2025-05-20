# Policy Iteration GridWorld Example

This project demonstrates the implementation of the **Policy Iteration** algorithm in a GridWorld environment using Python. The GridWorld is a simple 4x4 grid where an agent navigates from a starting position to a goal position while avoiding obstacles (holes) and maximizing rewards.

---

## 📂 Project Structure

- **`grid.py`**: Contains the `GridWorld` class, which defines the environment, including the grid layout, actions, and utility methods.
- **`policy_iteration.py`**: Implements the Policy Iteration algorithm to solve the GridWorld environment.
- **`test_grid.py`**: Unit tests for the `GridWorld` class to ensure correctness of the environment setup and methods.
- **`README.md`**: This file, providing an overview of the project.

---

## 🗺️ Environment Description

- **Grid Layout**:
  - `s`: Start position of the agent.
  - `g`: Goal position the agent must reach.
  - `h`: Holes (obstacles) that the agent must avoid.
  - `.`: Empty spaces where the agent can move.

- **Actions**:
  - `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`.

- **Rewards**:
  - `+10`: Reaching the goal.
  - `-1`: Each step taken.
  - `-5`: Falling into a hole.

---

## 🚀 How It Works

1. **GridWorld Initialization**:
   - The environment is initialized with a 4x4 grid.
   - The agent starts at the `s` position and must navigate to the `g` position.

2. **Policy Iteration**:
   - **Policy Evaluation**: Computes the value of each state under the current policy.
   - **Policy Improvement**: Updates the policy by selecting the best action for each state.
   - Repeats until the policy converges to the optimal policy.

3. **Output**:
   - The optimal policy and value function for the GridWorld environment.

---

## 🛠️ How to Run

1. **Install Dependencies**:
   Ensure you have Python 3.x and NumPy installed:
   ```bash
   pip install numpy

2. **Run the Policy Iteration Algorithm**: 
    Execute the policy_iteration.py script:
    '''bash
    python policy_iteration.py

3. **Run Tests**:
    To verify the environemtn setup, run the tests:
    '''bash
    python -m unittest test_grid.py


## 📊 Results
The program outputs:

The optimal policy for each state in the grid.
The value function for each state.

## 📝 Notes
The environment is customizable. You can modify the grid layout, rewards, and transition dynamics in grid.py.
This project is a foundational example for understanding reinforcement learning concepts like Policy Iteration.