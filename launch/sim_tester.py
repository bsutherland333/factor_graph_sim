import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../src'))
sys.path.append(src_dir)

from landmarks import generate_uniform_random_landmarks
from plotter import plot_field
from path import *
from measurements import generate_gaussian_measurements

import numpy as np

field_range = np.array([[0, 10], [0, 10]])

landmarks = generate_uniform_random_landmarks(15, field_range)
path = arc_path(50, np.array([0, 0]), np.array([10, 10]), 30)
noisy_path = path + np.random.normal(0, 0.05, path.shape)
measurements, measurement_associations = generate_gaussian_measurements(path, landmarks, range_std=0.05, bearing_std=0.05, max_range=4)

plot_field(landmarks=landmarks, estimated_poses=noisy_path, true_poses=path,
           measurement_associations=measurement_associations)

