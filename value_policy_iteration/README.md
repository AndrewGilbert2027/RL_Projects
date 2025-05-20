
Where:
- `V(s)` is the value of state `s`.
- `R(s, a)` is the reward for taking action `a` in state `s`.
- `P(s'|s, a)` is the transition probability to state `s'` from state `s` after action `a`.
- `γ` (gamma) is the discount factor.

The algorithm stops when the value function converges, and the optimal policy can be derived by selecting the action that maximizes the value for each state.

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