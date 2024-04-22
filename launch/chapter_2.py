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
import time


# Set random seed for reproducibility
np.random.seed(11)


# Noise/bias parameters
measurement_range_std = 0.1
measurement_bearing_std = 0.05
odometry_range_bias = 0.05
odometry_range_std = 0.05
odometry_angle_bias = 0.05
odometry_angle_std = 0.05


# Generate simulated information for solver
field_range = np.array([[0, 10], [0, 10]])
landmarks = generate_uniform_random_landmarks(15, field_range)
path = arc_path(20, np.array([0, 0]), np.array([10, 10]), 30)
measurements, measurement_associations = \
        generate_gaussian_measurements(path, landmarks, range_std=measurement_range_std,
                                       bearing_std=measurement_bearing_std, max_range=4)
odometry, odometry_path = generate_odometry(path, range_std=odometry_range_std,
                                            angle_std=odometry_angle_std,
                                            range_bias=odometry_range_bias,
                                            angle_bias=odometry_angle_bias)


# Find the linearized least squares problem
measurement_poses = odometry_path[measurement_associations[:, 0]]
measurement_landmarks = landmarks[measurement_associations[:, 1]]
J_ranges = range_to_location_jacobian(measurement_poses, measurement_landmarks)
A = np.zeros((measurements.shape[0], odometry_path.shape[0] * (odometry_path.shape[1] - 1)))
for i, j in enumerate(measurement_associations[:, 0]):
    pose_idx = j * 2
    A[i, pose_idx:pose_idx + 2] = J_ranges[i, 0, :2]
A = A / measurement_range_std
b = (measurements[:, 0] - range_to_location(measurement_poses, measurement_landmarks)).reshape(-1, 1) / measurement_range_std


# Solve the least squares problem
delta = np.linalg.solve(A.T @ A, A.T @ b).reshape(-1, 2)
x = odometry_path[:, :2] + delta


# Plot the results
plot_field(landmarks=landmarks, true_poses=path, estimated_poses=x,
           measurement_associations=measurement_associations)

