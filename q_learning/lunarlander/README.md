# Q-Learning Techniques for Blackjack

This repository explores various Q-learning techniques applied to the Blackjack environment in [Gymnasium](https://gymnasium.farama.org/). The goal is to maximize rewards by learning an optimal policy for playing Blackjack.

## Environment

The Blackjack environment is a card game where the agent must learn to make decisions (hit or stick) based on the current state of the game. The state space includes the player's current hand value, the dealer's visible card, and whether the player has a usable ace. The action space consists of two discrete actions: `hit` (draw another card) or `stick` (stop drawing cards).

## Techniques Explored

### 1. Basic Q-Learning
- **Description**: Implements a simple Q-learning algorithm with a discrete state-action space.
- **Features**:
    - Epsilon-greedy exploration strategy.
    - Learning rate and discount factor tuning.

### 2. Double Q-Learning
- **Description**: Addresses overestimation bias in Q-learning by maintaining two separate Q-tables.
- **Features**:
    - Alternating updates between two Q-tables.
    - Improved stability and performance.

### 3. SARSA (State-Action-Reward-State-Action)
- **Description**: On-policy temporal difference learning algorithm.
- **Features**:
    - Learns the Q-value based on the action taken by the current policy.
    - Balances exploration and exploitation.

### 4. TD(N) 
- **Description**: Unifies Monte Carlo learning and Temporal Difference learning by sampling N time steps ahead.
- **Features**:
    - Samples the Rewards N steps ahead and then estimates using the bootstrapping method
    - Helps increase sample efficiency while also learning in realtime. 

### 5. Deep Q-Learning (DQN)
- **Description**: Extends Q-learning using a neural network to approximate the Q-function.
- **Features**:
    - Experience replay for efficient learning.
    - Target network to stabilize training.
    - Exploration decay using epsilon annealing.

## Installation

1. Clone the repository:
     ```bash
     git clone https://github.com/your-username/RL_Projects.git
     cd RL_Projects/q_learning/blackjack
     ```

2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

Run the desired Q-learning technique:
```bash
python main.py --algorithm <algorithm_name>
```
Replace `<algorithm_name>` with one of the following:
- `basic_q_learning`
- `double_q_learning`
- `sarsa`
- `dqn`

## Results

Each technique is evaluated based on:
- Average reward over episodes.
- Convergence speed.
- Stability of learning.

Detailed results and visualizations are available in the `results/` directory.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.