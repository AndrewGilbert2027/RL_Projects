#ifndef TRAIN_UTIL_H
#define TRAIN_UTIL_H
#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "grid.h"

// Function to initialize the Q-table with zeros
// The Q-table is a 2D vector where each row represents a state and each column represents an action
std::vector<std::vector<double>> get_q_table(int rows, int cols, int num_actions) {
    std::vector<std::vector<double>> q_table(rows * cols, std::vector<double>(num_actions, 0.0));
    return q_table;
}

// Returns the Action associated with the maximum Q-value for a given state
Actions get_greedy_action(const std::vector<std::vector<double>>& q_table, int state) {
    double max_value = q_table[state][0];
    Actions best_action = static_cast<Actions>(0);
    for (int action = 1; action < NUM_ACTIONS; ++action) {
        if (q_table[state][action] > max_value) {
            max_value = q_table[state][action];
            best_action = static_cast<Actions>(action);
        }
    }
    return best_action;
}

// Chooses greedy option (max) with a probability of (1 - epsilon)
// and a random action with a probability of epsilon
// This is the epsilon-greedy strategy for action selection
Actions get_epsilon_greedy_action(const std::vector<std::vector<double>>& q_table, int state, double epsilon) {
    if (static_cast<double>(rand()) / RAND_MAX < epsilon) {
        return static_cast<Actions>(rand() % NUM_ACTIONS);
    } else {
        return get_greedy_action(q_table, state);
    }
}


// Updates the Q-value for a given state-action pair using the Q-learning update rule
// The Q-learning update rule is: Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
void update_q_table(std::vector<std::vector<double>>& q_table, int state, Actions action, double reward, int next_state, double alpha, double gamma) {
    double max_next_q = *std::max_element(q_table[next_state].begin(), q_table[next_state].end());
    q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action]);
}




#endif // TRAIN_UTIL_H