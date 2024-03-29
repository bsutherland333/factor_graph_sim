import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from simulation.landmarks import generate_uniform_random_landmarks
from simulation.plotter import plot_field
from simulation.path import line_path, arc_path
from simulation.measurements import generate_gaussian_measurements

import numpy as np

field_range = np.array([[0, 10], [0, 10]])

landmarks = generate_uniform_random_landmarks(15, field_range)
path = arc_path(50, np.array([0, 0]), np.array([10, 10]), 30)
measurements, measurement_associations = generate_gaussian_measurements(path, landmarks, range_std=0.05, bearing_std=0.05, max_range=4)

plot_field(landmarks=landmarks, estimated_poses=path,
           measurement_associations=measurement_associations)

