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
            range_measurement = range_from_landmark(pose[i], landmarks[j])

            if range_measurement <= max_range:
                bearing_measurement = bearing_from_landmark(pose[i], landmarks[j])

                # Add noise to range and bearing measurements
                noisy_range = range_measurement + np.random.normal(0, range_std)
                noisy_bearing = bearing_measurement + np.random.normal(0, bearing_std)

                # Append the measurement as [range, bearing] to the measurements list
                measurements.append([noisy_range, noisy_bearing])
                associations.append([i, j])  # [pose_index, landmark_index]

    return np.array(measurements), np.array(associations)


def range_from_landmark(pose, landmark):
    """
    Calculate the range from a robot's pose to a landmark.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [x, y]
    landmark (np.array): The location of the landmark, in meters. [x, y]

    Returns:
    float: The range from the robot's pose to the landmark.
    """
    delta_x = landmark[0] - pose[0]
    delta_y = landmark[1] - pose[1]
    return np.sqrt(delta_x ** 2 + delta_y ** 2)


def bearing_from_landmark(pose, landmark):
    """
    Calculate the bearing from a robot's pose to a landmark.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [x, y]
    landmark (np.array): The location of the landmark, in meters. [x, y]

    Returns:
    float: The bearing from the robot's pose to the landmark.
    """
    delta_x = landmark[0] - pose[0]
    delta_y = landmark[1] - pose[1]
    return np.arctan2(delta_y, delta_x)
