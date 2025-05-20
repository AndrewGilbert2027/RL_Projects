#ifndef GRID_H
#define GRID_H
#include <iostream>
#include <cstdlib>


// Struct to return information about moving and the agent's state
struct Information {
    int state;
    int reward;
    bool terminated;
};

// Enum to represent the possible actions
// The agent can take in the grid world
enum Actions {
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,
    STAY = 4,
    NUM_ACTIONS = 5
};

// Enum to represent the possible rewards the agent can receive
enum Rewards {
    GOAL_REWARD = 10,
    STEP_PENALTY = -1,
    INVALID_MOVE_PENALTY = -5
};

// Class to represent the grid world
class Grid {

    public:
        Grid(int rows, int cols, int start_row, int start_col, int goal_row, int goal_col);
        ~Grid();

        Information move(Actions action);
        
        void reset_grid();
        
    private:
        int rows;
        int cols;
        char **grid;
        int start_row;
        int start_col;
        int goal_row;
        int goal_col;
        int current_row;
        int current_col;
        int step_count;
        int max_steps;
        int state_space;
        bool is_valid_move(int row, int col);
        void print_grid();
};


#endif // GRID_H