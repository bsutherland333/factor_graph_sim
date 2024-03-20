import numpy as np


def line_path(n_points, start_location, end_location):
    """
    Generates a straight line path.

    Parameters:
    n_points (int): The number of points to generate.
    start_location (np.array): The starting location of the line. [x, y]
    end_location (np.array): The ending location of the line. [x, y]

    Returns:
    np.array: The position and heading of the robot along a line path. [[x1, y1, theta1], ... ]
    """
    position = np.array([start_location + i * (end_location - start_location) / (n_points - 1)
                         for i in range(n_points)])
    heading = np.arctan2(end_location[1] - start_location[1],
                         end_location[0] - start_location[0]) * np.ones(n_points)
    return np.column_stack([position, heading])

def arc_path(n_points, start_location, end_location, angle):
    """
    Generates an arc path.

    Parameters:
    n_points (int): The number of points to generate.
    start_location (np.array): The starting location of the arc. [x, y]
    end_location (np.array): The ending location of the arc. [x, y]
    angle (float): The angle of the arc in degrees at the starting point, measured from the 
        line between the start and end points.

    Returns:
    np.array: The position and heading of the robot along an arc path. [[x1, y1, theta1], ... ]
    """

    # Clamp the angle to 0-90 degrees
    if angle >= 90 or angle < 0:
        print('Warning: Arc angles greater than or equal to 90 degrees or less than 0',
              'are not supported by arc_path, clamping...')
        angle = np.clip(angle, 0, 89.99999)

    # If the angle is 0, generate a line path to avoid numerical instability
    if angle == 0:
        return line_path(n_points, start_location, end_location)

    angle = np.deg2rad(angle)

    # Calculate the center and radius of the arc
    radius = np.linalg.norm(end_location - start_location) / (2 * np.sin(angle))
    angle_from_zero = np.arctan2(end_location[1] - start_location[1],
                                 end_location[0] - start_location[0]) + angle
    center_point = start_location + radius * np.array([np.cos(angle_from_zero - np.pi / 2),
                                                       np.sin(angle_from_zero - np.pi / 2)])

    # Calculate the start and end angles
    start_angle = np.arctan2(start_location[1] - center_point[1],
                             start_location[0] - center_point[0])
    end_angle = np.arctan2(end_location[1] - center_point[1],
                           end_location[0] - center_point[0])

    # Calculate the delta angle per point
    if np.abs(start_angle - end_angle) > np.pi:
        delta_angle = -(np.pi*2 - (end_angle - start_angle)) / (n_points - 1)
    else:
        delta_angle = (end_angle - start_angle) / (n_points - 1)

    # Generate the points
    points = np.array([center_point + radius * np.array([np.cos(start_angle + i * delta_angle),
                                                         np.sin(start_angle + i * delta_angle)])
                       for i in range(n_points)])
    heading = np.array([angle_from_zero + i * delta_angle for i in range(n_points)])
    heading -= np.floor((heading + np.pi) / (2*np.pi)) * 2*np.pi

    return np.column_stack([points, heading])
