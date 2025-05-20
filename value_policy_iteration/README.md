# Value and Policy Iteration Algorithms

Value Iteration and Policy Iteration are two fundamental algorithms in reinforcement learning used to solve Markov Decision Processes (MDPs). These algorithms aim to find an optimal policy that maximizes the cumulative reward for an agent interacting with an environment.

---

## 📘 Value Iteration

Value Iteration is an iterative algorithm that computes the optimal value function for each state. It works by repeatedly updating the value of each state based on the Bellman optimality equation:

$V(s) = max_a [ R(s, a) + γ * Σ P(s'|s, a) * V(s') ]$

Where:
- `V(s)` is the value of state `s`.
- `R(s, a)` is the mean reward for taking action `a` in state `s`.
- `P(s'|s, a)` is the transition probability to state `s'` from state `s` after action `a`.
- `γ` (gamma) is the discount factor.

The algorithm stops when the value function converges, and the optimal policy can be derived by selecting the action that maximizes the value for each state.

In our update step, we have

$V_{k+1} = max_{\pi}[r_{\pi} + \gamma P_{\pi}V_{k}]$

By the contraction mapping theorem, we are certain that this iterative algorithm converges to the optimal policy.

***Remarks***:
1. R(s, a) and P(s'|s, a) are already given
2. V(s) and V(s') are to be calculated

***Uses***:
One should remember that Value and Policy Iteration algorithms should only be used when the given game or process is already known. In Reinforcement learning jargon, this means that we have a 'model' of the environement. If a model is not given (R(s, a) and P(s'|s, a) is not known) then we must turn to different methods. 

---

## 📘 Policy Iteration

Policy Iteration alternates between two steps to find the optimal policy:

1. **Policy Evaluation**:
   - Compute the value function for a given policy by solving the Bellman equation iteratively or directly.

2. **Policy Improvement**:
   - Update the policy by choosing the action that maximizes the value function for each state.

The process repeats until the policy stabilizes (i.e., no further changes occur), resulting in the optimal policy.

---

## 🔑 Key Differences

- **Value Iteration** focuses on directly finding the optimal value function and derives the policy afterward.
- **Policy Iteration** alternates between evaluating a policy and improving it until convergence.

Both algorithms are widely used in reinforcement learning and serve as the foundation for more advanced methods like Q-learning and Deep Reinforcement Learning.

--- 

These algorithms are essential for understanding how agents can learn to make optimal decisions in uncertain environments.
You can find more information on these algorithms [here](https://www.youtube.com/watch?v=wMAVmLDIvQU&list=PLEhdbSEZZbDaFWPX4gehhwB9vJZJ1DNm8&index=13)
