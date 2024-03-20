import numpy as np


class Field:
    """
    Generates a field with landmarks for the robot to explore.
    """

    def __init__(self, field_range, n_landmarks=10, landmark_pattern='random'):
        """
        Initializes the field with the given parameters.

        Parameters:
        field_range (list): The range of the field. [x_range, y_range]
        n_landmarks (int): The number of landmarks to generate.
        landmark_pattern (str): The pattern to generate the landmarks.
        """
        self.n_landmarks = n_landmarks
        self.field_range = field_range
        self.landmark_pattern = landmark_pattern
        self.landmarks = self._generate_landmarks()

    def _generate_landmarks(self):
        """
        Generates the landmarks for the field.
        """
        if self.landmark_pattern == 'random':
            return np.random.rand(self.n_landmarks, 2) \
                    * np.array([self.field_range[0], self.field_range[1]])
        else:
            raise NotImplementedError('Landmark pattern not implemented.')

    def get_absolute_landmarks(self):
        """
        Returns the absolute positions of the landmarks.

        Returns:
        np.array: The absolute positions of the landmarks. [[x1, y1], [x2, y2], ...]
        """
        return self.landmarks

    def get_relative_landmarks(self, robot_position, robot_heading):
        """
        Returns the relative positions of the landmarks from the robot's perspective.

        Parameters:
        robot_position (np.array): The position of the robot in the field. [x, y]
        robot_heading (float): The heading of the robot in radians.

        Returns:
        np.array: The relative positions of the landmarks. [[x1, y1], [x2, y2], ...]
        """
        relative_landmarks = self.landmarks - robot_position
        rotation_matrix = np.array([[np.cos(robot_heading), -np.sin(robot_heading)],
                                    [np.sin(robot_heading), np.cos(robot_heading)]])
        return np.dot(relative_landmarks, rotation_matrix)

    def get_field_range(self):
        """
        Returns the x and y range of the field.

        Returns:
        np.array: The x and y range of the field. [[x_min, x_max], [y_min, y_max]]
        """
        return np.array([[0, self.field_range[0]],
                         [0, self.field_range[1]]])

