import numpy as np

state_bins = [
    np.linspace(-1.0, 1.0, 10),  # Horizontal position
    np.linspace(-1.0, 1.0, 10),  # Vertical position
    np.linspace(-1.0, 1.0, 10),  # Horizontal velocity
    np.linspace(-1.0, 1.0, 10),  # Vertical velocity
    np.linspace(-np.pi, np.pi, 10),  # Angle
    np.linspace(-5.0, 5.0, 10),   # Angular velocity
    np.array([0, 1]),               # Left leg contact (binary)
    np.array([0, 1])                # Right leg contact (binary)
]

def get_discrete_state(state):
    """
    Convert a continuous state into a discrete state based on predefined bins.
    :param state: Continuous state from the environment.
    :return: Discrete state as a tuple of indices corresponding to the bins.
    """
    return tuple(np.digitize(s, b) for s, b in zip(state, state_bins))

def test_get_discrete_state():
    """
    Test the get_discrete_state function with various continuous states.
    """
    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]),
        np.array([-1.0, -1.0, -1.0, -1.0, -np.pi, -5.0, 1, 1]),
        np.array([1.0, 1.0, 1.0, 1.0, np.pi, 5.0, 0, 1]),
        np.array([0.5, -0.5, 0.5, -0.5, np.pi/2, -2.5, 1, 0])
    ]

    for state in test_states:
        discrete_state = get_discrete_state(state)
        print(f"Continuous state: {state}, Discrete state: {discrete_state}")

if __name__ == "__main__":
    test_get_discrete_state()
    # Example usage
    example_state = np.array([0.2, -0.3, 0.1, 0.4, np.pi/4, -1.0, 0, 1])
    discrete_state = get_discrete_state(example_state)
    print(f"Example continuous state: {example_state}, Discrete state: {discrete_state}")