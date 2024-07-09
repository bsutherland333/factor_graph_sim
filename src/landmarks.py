import numpy as np


def generate_uniform_random_landmarks(n_landmarks, field_range):
    """
    Generates a set of random landmarks uniformly distributed across the field (non-Gaussian).

    Parameters:
    n_landmarks (int): The number of landmarks to generate.
    field_range (list): The range of the field. [[x_min, x_max], [y_min, y_max]]

    Returns:
    np.array: The landmarks. [[x1, y1], [x2, y2], ...]
    """
    return np.random.uniform(field_range[0][0], field_range[0][1], (n_landmarks, 2))
