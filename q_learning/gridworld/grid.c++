#include "grid.h" // Include the header file for the Grid class
#include <iostream> // Include iostream for input/output operations

// Constructor: Initializes the grid with given dimensions, start, and goal positions
Grid::Grid(int rows, int cols, int start_row, int start_col, int goal_row, int goal_col) {
    this->rows = rows; // Number of rows in the grid
    this->cols = cols; // Number of columns in the grid
    this->start_row = start_row; // Starting row for the agent
    this->start_col = start_col; // Starting column for the agent
    this->goal_row = goal_row; // Goal row position
    this->goal_col = goal_col; // Goal column position
    this->current_row = start_row; // Current row of the agent
    this->current_col = start_col; // Current column of the agent
    this->step_count = 0; // Counter for the number of steps taken
    this->max_steps = 100; // Maximum allowed steps before termination
    this->state_space = rows * cols; // Total number of states in the grid

    // Allocate memory for the grid and initialize it with '.'
    grid = new char*[rows];
    for (int i = 0; i < rows; i++) {
        grid[i] = new char[cols];
        for (int j = 0; j < cols; j++) {
            grid[i][j] = '.'; // Empty cell
        }
    }
    grid[goal_row][goal_col] = 'G'; // Mark the goal position
}

// Destructor: Frees the allocated memory for the grid
Grid::~Grid() {
    for (int i = 0; i < rows; i++) {
        delete[] grid[i]; // Free each row
    }
    delete[] grid; // Free the grid array
}

// Resets the grid to its initial state
void Grid::reset_grid() {
    current_row = start_row; // Reset agent's row position
    current_col = start_col; // Reset agent's column position
    step_count = 0; // Reset step counter
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grid[i][j] = '.'; // Clear the grid
        }
    }
    grid[goal_row][goal_col] = 'G'; // Re-mark the goal position
}

// Prints the current state of the grid
void Grid::print_grid() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == current_row && j == current_col) {
                std::cout << 'A'; // Display agent's position
            } else {
                std::cout << grid[i][j]; // Display grid cell
            }
        }
        std::cout << std::endl; // New line after each row
    }
}

// Checks if a move to the specified position is valid
bool Grid::is_valid_move(int row, int col) {
    return (row >= 0 && row < rows && col >= 0 && col < cols); // Within bounds
}

// Handles the agent's movement and returns information about the new state
Information Grid::move(Actions action) {
    Information info;
    info.state = current_row * cols + current_col; // Current state index
    info.terminated = false; // Whether the episode has terminated
    info.reward = STEP_PENALTY; // Default step penalty

    int new_row = current_row; // Temporary row for the new position
    int new_col = current_col; // Temporary column for the new position

    // Update position based on the action
    switch (action) {
        case UP:
            new_row--;
            break;
        case DOWN:
            new_row++;
            break;
        case LEFT:
            new_col--;
            break;
        case RIGHT:
            new_col++;
            break;
        case STAY:
            break; // No movement
        default:
            std::cerr << "Invalid action" << std::endl;
            return info; // Return unchanged info for invalid action
    }

    // Check if the move is valid
    if (is_valid_move(new_row, new_col)) {
        current_row = new_row; // Update agent's row position
        current_col = new_col; // Update agent's column position

        // Check if the agent reached the goal
        if (current_row == goal_row && current_col == goal_col) {
            info.reward = GOAL_REWARD; // Reward for reaching the goal
            info.terminated = true; // Terminate the episode
        } else {
            info.reward = STEP_PENALTY; // Regular step penalty
        }
        info.state = current_row * cols + current_col; // Update state index
    } else {
        info.reward = INVALID_MOVE_PENALTY; // Penalty for invalid move
    }

    step_count++; // Increment step counter
    if (step_count >= max_steps) {
        info.terminated = true; // Terminate if max steps reached
    }

    return info; // Return updated information
}