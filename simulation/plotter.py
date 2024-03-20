import matplotlib.pyplot as plt
import numpy as np

def plot_field(landmarks=None, estimated_poses=None, true_poses=None, 
               measurements=None, title=None):
    """
    Plot the field with the landmarks, estimated_poses, and measurements.

    Parameters:
    landmarks (np.array): The absolute positions of the landmarks. [[x1, y1], ...]
    estimated_poses (np.array): The positions of the robot's estimated poses. [[x1, y1], ...]
    true_poses (np.array): The positions of the robot's true poses. [[x1, y1], ...]
    measurements (list): The measurements of the robot. [[landmark_index, pose_index], ...]
    title (str): The title of the plot.
    """
    # Plot measurements
    if measurements is not None and estimated_poses is not None and landmarks is not None:
        for measurement in measurements:
            landmark_index, pose_index = measurement
            landmark = landmarks[landmark_index]
            pose = estimated_poses[pose_index]
            plt.plot([pose[0], landmark[0]], [pose[1], landmark[1]], color='grey', linestyle='--',
                     label='Measurements' if pose_index == 0 else None)

    # Plot landmarks
    if landmarks is not None:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], color='grey', label='Landmarks', s=60)

    # Plot robot true_poses
    if true_poses is not None:
        plt.plot(true_poses[:, 0], true_poses[:, 1], marker='o', color='red', label='True Pose')

    # Plot robot estimated_poses
    if estimated_poses is not None:
        plt.plot(estimated_poses[:, 0], estimated_poses[:, 1], marker='o',
                 color='blue', label='Estimated Pose')

    # Add title and legend
    if title is not None:
        plt.title(title)
    plt.legend()

    # Set scale to be equivalent
    plt.axis('equal')

    # Show plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

