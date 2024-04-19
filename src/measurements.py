import numpy as np

def generate_gaussian_measurements(pose, landmarks, range_std, bearing_std, max_range):
    """
    Generates range and bearing measurements for the robot. A measurement for every landmark is
    generated for each pose if the landmark is within the maximum range of the sensor.
    Measurements are in body frame.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [[x1, y1, psi1], ...]
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
    pose (np.array): The robot's pose, in meters and radians. [x, y, psi]
    landmark (np.array): The location of the landmark, in meters. [x, y]

    Returns:
    float: The range from the robot's pose to the landmark.
    """
    delta_x = landmark[0] - pose[0]
    delta_y = landmark[1] - pose[1]
    return np.sqrt(delta_x ** 2 + delta_y ** 2)


def bearing_from_landmark(pose, landmark):
    """
    Calculate the relative bearing from a robot's pose to a landmark. Meaning, if the robot is
    facing directly at the landmark, the bearing will be 0 (body frame).

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [x, y, psi]
    landmark (np.array): The location of the landmark, in meters. [x, y]

    Returns:
    float: The bearing from the robot's pose to the landmark.
    """
    delta_x = landmark[0] - pose[0]
    delta_y = landmark[1] - pose[1]
    return np.arctan2(delta_y, delta_x) - pose[2]


def generate_odometry(path, range_noise, angle_noise, range_bias, angle_bias):
    """
    Generates odometry measurements for a robot following a path. The measurements are assumed
    to have gaussian noise and a bias in both the distance and angle. Angle measurements returned
    are based on the change in heading in body frame.

    Parameters:
    path (np.array): The robot's path, in meters and radians. [[x1, y1, psi1], ...]
    range_noise (float): The standard deviation of the range measurements, in meters.
    angle_noise (float): The standard deviation of the angle measurements, in radians.
    range_bias (float): The bias in the range measurements, in meters.
    angle_bias (float): The bias in the angle measurements, in radians.

    Returns:
    np.array: The odometry measurements for the robot. Each measurement is for the current point,
        specifying the odometry to the next point. As such, there will be no measurement for the
        last point in the path. [[range1, angle1], ...]
    np.array: The path generated using the odometry measurements. This will in the same format as
        the original path, just with noise and bias applied. [[x1, y1, psi1], ...]
    """

    odometry = []

    # Generate measurements
    for i in range(path.shape[0] - 1):
        delta_x = path[i + 1, 0] - path[i, 0]
        delta_y = path[i + 1, 1] - path[i, 1]
        delta_psi = path[i + 1, 2] - path[i, 2]

        # Calculate the range and angle to the next point
        range_measurement = np.sqrt(delta_x ** 2 + delta_y ** 2) \
                + np.random.normal(0, range_noise) + range_bias
        angle_measurement = delta_psi + np.random.normal(0, angle_noise) + angle_bias

        odometry.append([range_measurement, angle_measurement])
    odometry = np.array(odometry)

    # Generate a path using the odometry
    odom_path = np.zeros_like(path)
    odom_path[0] = path[0]
    for i in range(1, path.shape[0]):
        odom_path[i] = get_next_pose_from_odom(odom_path[i - 1], odometry[i - 1])

    return odometry, odom_path

def get_next_pose_from_odom(pose, odometry):
    """
    Calculate the next pose of the robot given the current pose and odometry measurements.

    Parameters:
    pose (np.array): The robot's current pose, in meters and radians. [x, y, psi]
    odometry (np.array): The odometry measurements, in meters and radians. [range, angle]

    Returns:
    np.array: The robot's next pose, in meters and radians. [x, y, psi]
    """
    return np.array([pose[0] + odometry[0] * np.cos(pose[2] + odometry[1]*0.5),
                     pose[1] + odometry[0] * np.sin(pose[2] + odometry[1]*0.5),
                     pose[2] + odometry[1]])

