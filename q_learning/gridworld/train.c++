#include "grid.h"
#include "train_util.h"
#include <iostream>
#include "math.h"

int main() {
    // Initialize the grid world (assumes nxn grid)
    int rows = 5;
    int cols = 5;
    int start_row = 0;
    int start_col = 0;
    int goal_row = 4;
    int goal_col = 4;

    Grid grid(rows, cols, start_row, start_col, goal_row, goal_col);

    double alpha = 0.1; // Learning rate
    double gamma = 0.9; // Discount factor
    double epsilon = 1.0; // Exploration rate
    double min_epsilon = 0.02; // Minimum exploration rate
    double epsilon_decay = 0.99; // Decay rate for exploration
    int state = start_row * cols + start_col;
    double discounted_reward;

    std::vector<std::vector<double>> q_table = get_q_table(rows , cols, NUM_ACTIONS);
    std::vector<double> rewards(1000, 0.0); // Store the discounted rewards for each episode
    std::cout << "Training started..." << std::endl;

    // Training loop
    // The agent will train for 1000 episodes
    for (int episode = 0; episode < 1000; ++episode) {
        Information info;
        discounted_reward = 0.0;
        grid.reset_grid();
        int step_count = 0;

        while (true) {
            Actions action = get_epsilon_greedy_action(q_table, state, epsilon);
            info = grid.move(action);
            epsilon = std::max(min_epsilon, epsilon * epsilon_decay); // Decay epsilon (explores early and exploits later)

            int next_state = info.state;
            double reward = info.reward;
            discounted_reward += reward * pow(gamma, step_count);
            step_count++;

            update_q_table(q_table, state, action, reward, next_state, alpha, gamma);
            state = next_state;

            if (info.terminated) {
                rewards[episode] = discounted_reward;
                std::cout << "Episode: " << episode << ", Discounted Reward: " << discounted_reward << std::endl;
                break;
            }
        }
    }

    std::cout << "Training completed." << std::endl;

    return 0;
}