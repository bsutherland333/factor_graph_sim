import numpy as np

def generate_gaussian_measurements(pose, landmarks, range_std, bearing_std, max_range):
    """
    Generates range and bearing measurements for the robot. A measurement for every landmark is
    generated for each pose if the landmark is within the maximum range of the sensor.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [[x1, y1, theta1], ...]
    landmarks (np.array): The locations of the landmarks, in meters. [[x1, y1], [x2, y2], ...]
    range_std (float): The standard deviation of the range measurements.
    bearing_std (float): The standard deviation of the bearing measurements.
    max_range (float): The maximum range of the sensor, in meters.

    Returns:
    np.array: The range and bearing measurements for the robot. [[range1, bearing1], ...]
    np.array: The measurement associations. [[pose_index, landmark_index], ...]
    """
    measurements = []
    associations = []

    for i in range(pose.shape[0]):
        for j in range(landmarks.shape[0]):
            # Calculate the range and bearing between the robot's pose and the landmark
            delta_x = landmarks[j, 0] - pose[i, 0]
            delta_y = landmarks[j, 1] - pose[i, 1]
            range_to_landmark = np.sqrt(delta_x ** 2 + delta_y ** 2)

            if range_to_landmark <= max_range:
                # Add noise to range and bearing measurements
                noisy_range = range_to_landmark + np.random.normal(0, range_std)
                bearing = np.arctan2(delta_y, delta_x) - pose[i, 2]
                noisy_bearing = bearing + np.random.normal(0, bearing_std)

                # Append the measurement as [range, bearing] to the measurements list
                measurements.append([noisy_range, noisy_bearing])
                associations.append([i, j])

    return np.array(measurements), np.array(associations)

