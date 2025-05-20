# Reinforcement Learning Playground

Welcome to the **Reinforcement Learning Playground** – a modular collection of implementations, experiments, and environments designed to explore a wide range of RL algorithms in isolation. This project is organized by **algorithmic technique** (e.g., Q-Learning, Deep Q-Learning, Monte Carlo, TD(λ)) and demonstrates each with one or more game or simulation environments.

---

## 🗂️ Project Structure

The repository is structured as follows:
reinforcement-learning-project/
│
├── q_learning/
│ ├── gridworld/
│ └── cliff_walking/
│
├── deep_q_learning/
│ └── cartpole/
│
├── monte_carlo/
│ └── blackjack/
│
├── td_lambda/
│ └── random_walk/
│
├── function_approximation/
│ └── mountain_car/
│
├── README.md
└── requirements.txt



Each algorithm (top-level folder) contains subdirectories for individual environments. These environments are designed to be self-contained, with their own training logic, models, and README files.

---

## 📌 Key Principles

- **Modularity**: Each environment can be run and modified independently.
- **Clarity**: Minimal dependencies per folder, clear scripts, and documentation.
- **Reproducibility**: Each subfolder includes instructions to run experiments.
- **Isolation**: No shared state across different folders — you can delete one without affecting others.

---

## 🚀 Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AndrewGilbert2027/RL_Projects.git
   cd reinforcement-learning-project


