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
            range_measurement = range_to_location(pose[i].reshape(-1, 3),
                                                  landmarks[j].reshape(-1, 2))[0]

            if range_measurement <= max_range:
                bearing_measurement = bearing_to_location(pose[i].reshape(-1, 3),
                                                          landmarks[j].reshape(-1, 2))[0]

                # Add noise to range and bearing measurements
                noisy_range = range_measurement + np.random.normal(0, range_std)
                noisy_bearing = bearing_measurement + np.random.normal(0, bearing_std)

                # Append the measurement as [range, bearing] to the measurements list
                measurements.append([noisy_range, noisy_bearing])
                associations.append([i, j])  # [pose_index, landmark_index]

    return np.array(measurements), np.array(associations)


def range_to_location(pose, location):
    """
    Calculate the range from a robot's pose to a location of interest.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [[x1, y1, psi1], ...]
    location (np.array): The location of interest, in meters. [[x1, y1], ...]

    Returns:
    np.array: The range from the robot's pose to the location. [range1, ...]
    """

    return np.linalg.norm(location - pose[:, :2], axis=1)


def range_to_location_jacobian(pose, location):
    """
    Calculates the jacobian of the range measurement from a robot's pose to a location of interest,
    with respect to and evaluated at the robot's pose.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [[x1, y1, psi1], ...]
    location (np.array): The location of interest, in meters. [[x1, y1], ...]

    Returns:
    np.array: The jacobian of the range. [[[d_range/d_x, d_range/d_y, d_range/d_psi]], ...]
    """

    delta_x = pose[:, 0] - location[:, 0]
    delta_y = pose[:, 1] - location[:, 1]
    range_ = range_to_location(pose, location)

    d_range_d_x = delta_x / range_
    d_range_d_y = delta_y / range_
    d_range_d_psi = np.zeros_like(range_)

    return np.array([[d_range_d_x, d_range_d_y, d_range_d_psi]]).T.swapaxes(1, 2)


def bearing_to_location(pose, location):
    """
    Calculate the bearing from a robot's pose to a location of interest.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [[x1, y1, psi1], ...]
    location (np.array): The location of interest, in meters. [[x1, y1], ...]

    Returns:
    np.array: The bearing from the robot's pose to the location. [bearing1, ...]
    """

    delta_x = location[:, 0] - pose[:, 0]
    delta_y = location[:, 1] - pose[:, 1]
    return np.arctan2(delta_y, delta_x) - pose[:, 2]


def bearing_to_location_jacobian(pose, location):
    """
    Calculates the jacobian of the range and bearing measurements from a robot's pose to a location
    of interest, with respect to and evaluated at the robot's pose.

    Parameters:
    pose (np.array): The robot's pose, in meters and radians. [[x1, y1, psi1], ...]
    location (np.array): The location of interest, in meters. [[x1, y1], ...]

    Returns:
    np.array: The jacobian of the range and bearing. 
        [[[d_bearing/d_x, d_bearing/d_y, d_bearing/d_psi]], ...]
    """

    delta_x = pose[:, 0] - location[:, 0]
    delta_y = pose[:, 1] - location[:, 1]

    d_bearing_d_x = -delta_y / (delta_x**2 + delta_y**2)
    d_bearing_d_y = delta_x / (delta_x**2 + delta_y**2)
    d_bearing_d_psi = np.ones_like(delta_x) * -1

    return np.array([[d_bearing_d_x, d_bearing_d_y, d_bearing_d_psi]]).T.swapaxes(1, 2)


def generate_odometry(path, range_std, angle_std, range_bias, angle_bias):
    """
    Generates odometry measurements for a robot following a path. The measurements are assumed
    to have gaussian noise and a bias in both the distance and angle. Angle measurements returned
    are based on the change in heading in body frame, which is a relative measurement.

    Parameters:
    path (np.array): The robot's path, in meters and radians. [[x1, y1, psi1], ...]
    range_std (float): The standard deviation of the range measurements, in meters.
    angle_std (float): The standard deviation of the angle measurements, in radians.
    range_bias (float): The bias in the range measurements, in meters.
    angle_bias (float): The bias in the angle measurements, in radians.

    Returns:
    np.array: The odometry measurements for the robot. Each measurement is for the second point in
        a pair, specifying the odometry from the previous point. As such, there will be no
        measurement for the first point in the path. [[range1, angle1], ...]
    """

    odometry = np.array([range_to_location(path[:-1], path[1:, :2]),
                         path[1:, 2] - path[:-1, 2]]).T
    odometry[:, 0] += np.random.normal(0, range_std, odometry.shape[0]) + range_bias
    odometry[:, 1] += np.random.normal(0, angle_std, odometry.shape[0]) + angle_bias

    return odometry


def get_path_from_odometry(x0, odometry):
    """
    Generate a path from the initial pose and odometry measurements.

    Parameters:
    x0 (np.array): The initial pose of the robot, in meters and radians. [x, y, psi]
    odometry (np.array): The odometry measurements, in meters and radians. [[range1, angle1], ...]

    Returns:
    np.array: The path generated using the odometry measurements. [[x1, y1, psi1], ...]
    """

    path = np.zeros((odometry.shape[0] + 1, 3))
    path[0] = x0
    for i in range(1, path.shape[0]):
        path[i] = get_next_pose_from_odom(path[i - 1].reshape(-1, 3),
                                          odometry[i - 1].reshape(-1, 2))[0]

    return path


def get_next_pose_from_odom(pose, odometry):
    """
    Calculate the next pose of the robot given the current pose and next set odometry measurements.

    NOTE: This function assumes the robot will follow an arc inbetween poses. As long as each pose
    is close enough to the next then this should be a good assumption.

    Parameters:
    pose (np.array): The robot's current pose, in meters and radians. [[x1, y1, psi1], ...]
    odometry (np.array): The odometry measurements, in meters and radians. [[range1, angle1], ...]

    Returns:
    np.array: The robot's next pose, in meters and radians. [[x, y, psi], ...]
    """
    return np.array([pose[:, 0] + odometry[:, 0] * np.cos(pose[:, 2] + odometry[:, 1]*0.5),
                     pose[:, 1] + odometry[:, 0] * np.sin(pose[:, 2] + odometry[:, 1]*0.5),
                     pose[:, 2] + odometry[:, 1]]).T


def odom_range_jacobian(pose):
    """
    Calculates the jacobian of the range odometry measurements for each pair of poses.

    Parameters:
    pose (np.array): The robot's poses, in meters and radians. [[x1, y1, psi1], ...]

    Returns:
    np.array: The jacobian of the range odometry measurements. [[[d_range/d_x1, d_range/d_y1, d_range/d_psi1,
                                                                  d_range/d_x2, d_range/d_y2, d_range/d_psi2]], ...]
    """

    range = np.linalg.norm(pose[:-1, :2] - pose[1:, :2], axis=1)
    dr_dx1 = (pose[:-1, 0] - pose[1:, 0]) / range
    dr_dx2 = -dr_dx1
    dr_dy1 = (pose[:-1, 1] - pose[1:, 1]) / range
    dr_dy2 = -dr_dy1
    dr_psi = np.zeros_like(range)

    return np.array([[dr_dx1, dr_dy1, dr_psi, dr_dx2, dr_dy2, dr_psi]]).T.swapaxes(1, 2)


def odom_bearing_jacobian(pose):
    """
    Calculates the jacobian of the bearing odometry measurements for each pair of poses.

    This jacobian makes use of the assumption that the robot follows an arc between poses and that the heading of the
    robot will be tangent to this arc. This enables the change in heading to be depended on the change in position.
    This dependency allows for solving for poses of the robot that have no measurements, as long as the previous pose is
    known.

    Parameters:
    pose (np.array): The robot's poses, in meters and radians. [[x1, y1, psi1], ...]

    Returns:
    np.array: The jacobian of the bearing odometry measurements. [[[d_bearing/d_x1, d_bearing/d_y1, d_bearing/d_psi1],
                                                                   [d_bearing/d_x2, d_bearing/d_y2, d_bearing/d_psi2]],
                                                                   ...]
    """

    x1 = pose[:-1, 0]
    x2 = pose[1:, 0]
    y1 = pose[:-1, 1]
    y2 = pose[1:, 1]
    delta_x = x2 - x1
    delta_y = y2 - y1

    db_dx1 = -2*delta_y / (x1**2 + x2**2 - 2*x1*x2 + delta_y**2)
    db_dx2 = 2*delta_y / (x1**2 + x2**2 - 2*x1*x2 + delta_y**2)
    db_dy1 = 2*delta_x / (x1**2 + x2**2 - 2*x1*x2 + y1**2 + y2**2 - 2*y1*y2)
    db_dy2 = -2*delta_x / (x1**2 + x2**2 - 2*x1*x2 + y1**2 + y2**2 - 2*y1*y2)
    db_dpsi1 = np.zeros_like(delta_x)
    db_dpsi2 = np.ones_like(delta_x) * 2

    return np.array([[db_dx1, db_dy1, db_dpsi1, db_dx2, db_dy2, db_dpsi2]]).T.swapaxes(1, 2)
