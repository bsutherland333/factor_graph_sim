"""
This script constructs a basic SLAM-like problem where a robot moves through a field and collects
range and bearing measurements to known landmarks. The robot's pose is estimated by using scipy to
minimize a least squares cost function.

The problem is solved simultaneously rather than iteratively, no sparsity exploitation is used,
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
from non_sparse_solvers import *

import numpy as np
import time


np.random.seed(0)
np.set_printoptions(linewidth=np.inf, threshold=np.inf)

# Noise/bias parameters
measurement_range_std = 0.01    # m
measurement_bearing_std = 0.05  # rad
odometry_range_bias = 0.05      # m
odometry_range_std = 0.05       # m
odometry_angle_bias = 0.05      # rad
odometry_angle_std = 0.05       # rad

# Generate simulated information for solver
field_range = np.array([[0, 10], [0, 10]])
landmarks = generate_uniform_random_landmarks(15, field_range)
path = arc_path(20, np.array([0, 0]), np.array([10, 10]), 30)
measurements, measurement_associations = \
        generate_gaussian_measurements(path, landmarks, range_std=measurement_range_std,
                                       bearing_std=measurement_bearing_std, max_range=4)
odometry = generate_odometry(path, range_std=odometry_range_std,
                             angle_std=odometry_angle_std,
                             range_bias=odometry_range_bias,
                             angle_bias=odometry_angle_bias)
x0 = path[0]
x0_std = 1e-6
x = get_path_from_odometry(path[0], odometry)

plot_field(landmarks=landmarks, true_poses=path, estimated_poses=x[:, :2],
           measurement_associations=measurement_associations, title='Initial Odometry')

# Solve the problem
start_time = time.time()
for iter in range(50):
    # Find the linearized least-squares problem
    measurement_poses = x[measurement_associations[:, 0]]
    measurement_landmarks = landmarks[measurement_associations[:, 1]]
    J_ranges = range_to_location_jacobian(measurement_poses, measurement_landmarks)
    J_bearings = bearing_to_location_jacobian(measurement_poses, measurement_landmarks)
    J_odom_ranges = odom_range_jacobian(x)
    J_odom_bearings = odom_bearing_jacobian(x)

    # Construct the whitened Jacobian matrix
    pose_size = x.shape[1]
    num_measurements = measurements.shape[0]*2 + odometry.shape[0]*2 + pose_size
    num_states = x.shape[0] * pose_size
    A = np.zeros((num_measurements, num_states))
    for i, j in enumerate(measurement_associations[:, 0]):
        pose_idx = j * pose_size
        A[i, pose_idx:pose_idx + pose_size] = J_ranges[i, 0,] / measurement_range_std
        A[i + measurements.shape[0], pose_idx:pose_idx + pose_size] = \
                J_bearings[i, 0] / measurement_bearing_std
    for i in range(odometry.shape[0]):
        pose_idx = i * pose_size
        A[i + measurements.shape[0]*2, pose_idx:pose_idx + 2*pose_size] = \
                J_odom_ranges[i, 0] / odometry_range_std
        A[i + measurements.shape[0]*2 + odometry.shape[0], pose_idx:pose_idx + 2*pose_size] = \
                J_odom_bearings[i, 0] / odometry_angle_std
    A[-pose_size:, :pose_size] = -np.diag(np.ones(pose_size)) / x0_std

    # Construct the whitened residual vector
    b_ranges = (measurements[:, 0] - range_to_location(measurement_poses, \
            measurement_landmarks)).reshape(-1, 1) / measurement_range_std
    b_bearings = (measurements[:, 1] - bearing_to_location(measurement_poses, \
            measurement_landmarks)).reshape(-1, 1) / measurement_bearing_std
    expected_odometry = generate_odometry(x, 0, 0, 0, 0)
    b_odom_ranges = (odometry[:, 0] - expected_odometry[:, 0]).reshape(-1, 1) \
            / odometry_range_std
    b_odom_bearings = (odometry[:, 1] - expected_odometry[:, 1]).reshape(-1, 1) \
            / odometry_angle_std
    b_x0 = (x[0] - x0).reshape(-1, 1) / x0_std
    b = np.vstack((b_ranges, b_bearings, b_odom_ranges, b_odom_bearings, b_x0))

    # Solve the least squares problem
    x_prev = x.copy()
    x = cholesky_factorization(A, b, x)

    # Check for convergence
    if np.linalg.norm(x - x_prev) < 1e-3:
        break

# Print the time
end_time = time.time()
print(f'Final state: {x}')
print(f'Completed in {end_time - start_time}s, a rate of {1 / (end_time - start_time)}Hz')
print(f'Final error: {np.linalg.norm(path - x, axis=1).sum()}')


# Plot the results
plot_field(landmarks=landmarks, true_poses=path, estimated_poses=x[:, :2],
           measurement_associations=measurement_associations, title='Final estimate')

