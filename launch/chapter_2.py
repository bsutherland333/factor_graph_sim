"""
This script constructs a basic SLAM-like problem where a robot moves through a field and collects
range and bearing measurements to known landmarks. The robot's pose is estimated by using scipy to
minimize a least squares cost function.

The problem is solved simultaneously rather than iteratively, no sparsity explotation is used,
no special solvers are applied.
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '../src'))
sys.path.append(parent_dir)

from landmarks import generate_uniform_random_landmarks
from plotter import plot_field
from path import *
from measurements import *

import numpy as np
from scipy.optimize import minimize
import time

# Set random seed for reproducibility
np.random.seed(100)

# Generate simulated information for solver
field_range = np.array([[0, 10], [0, 10]])
landmarks = generate_uniform_random_landmarks(15, field_range)
path = arc_path(20, np.array([0, 0]), np.array([10, 10]), 30)
measurements, measurement_associations = generate_gaussian_measurements(path, landmarks, range_std=0.05, bearing_std=0.005, max_range=4)
odometry, odometry_path = generate_odometry(path, range_noise=0.03, angle_noise=0.003, range_bias=0.03, angle_bias=0.01)

# Define the cost function for the solver
def cost_function(x):
    """
    Calculate the least squares cost for the robot's pose given the measurements and landmarks.

    Parameters:
    x (np.array): The robot's poses, in meters. [x1, y1, psi1, x2, y2, psi2, ...]
    """

    # Reshape the poses into a 2D array
    x = x.reshape(-1, 3)

    cost = 0
    for i in range(measurements.shape[0]):
        landmark = landmarks[measurement_associations[i, 1]]
        pose = x[measurement_associations[i, 0]]
        range_measurement = measurements[i, 0]
        bearing_measurement = measurements[i, 1]

        expected_range = range_to_location(pose.reshape(-1, 3), landmark.reshape(-1, 2))[0]
        expected_bearing = bearing_to_location(pose.reshape(-1, 3), landmark.reshape(-1, 2))[0]

        cost += (range_measurement - expected_range)**2 \
                + (bearing_measurement - expected_bearing)**2

    for i in range(odometry.shape[0]):
        expected_pose = get_next_pose_from_odom(x[i].reshape(-1, 3), odometry[i].reshape(-1, 2))[0]
        cost += np.linalg.norm(x[i + 1] - expected_pose)**2

    return cost


# Solve the least squares problem with scipy
start_time = time.time()
initial_guess = odometry_path.flatten()
result = minimize(cost_function, initial_guess, method='CG')
runtime = time.time() - start_time
print(result)
print(f"Runtime: {runtime:.2f}s, Rate: {1 / runtime:.2f}Hz")


# Plot the results
x = result.x.reshape(-1, 3)
plot_field(landmarks=landmarks, true_poses=path, estimated_poses=x,
           measurement_associations=measurement_associations)

