import sys

import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def get_rgb_image_from_cv2(image_path, show=False):
    # Read image
    img = cv2.imread(image_path)
    img_rgb = None

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not loaded. Check the file path.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if show:
            # Display the image in a window named 'Image'
            cv2.imshow(image_path.split('.')[0], img)

            # Wait for a key press and close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return img_rgb

def pose_landmarks():
    # Dictionary mapping landmark indices to body parts
    landmark_dict = {
        0: 'nose',
        1: 'left eye (inner)',
        2: 'left eye',
        3: 'left eye (outer)',
        4: 'right eye (inner)',
        5: 'right eye',
        6: 'right eye (outer)',
        7: 'left ear',
        8: 'right ear',
        9: 'mouth (left)',
        10: 'mouth (right)',
        11: 'left shoulder',
        12: 'right shoulder',
        13: 'left elbow',
        14: 'right elbow',
        15: 'left wrist',
        16: 'right wrist',
        17: 'left pinky',
        18: 'right pinky',
        19: 'left index',
        20: 'right index',
        21: 'left thumb',
        22: 'right thumb',
        23: 'left hip',
        24: 'right hip',
        25: 'left knee',
        26: 'right knee',
        27: 'left ankle',
        28: 'right ankle',
        29: 'left heel',
        30: 'right heel',
        31: 'left foot index',
        32: 'right foot index'
    }

    return landmark_dict

def detect_pose_landmarks(image_path, pose=None, show=False):
    if pose is None:
        pose, mp_drawing, mp_pose = initialize_mediapipe()

    img_rgb = get_rgb_image_from_cv2(image_path, show=False)
    results = pose.process(img_rgb)

    landmarks_data = []
    if results.pose_landmarks:
        if show:
            # Extract landmark data
            body_parts = pose_landmarks()
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append({
                    'Landmark': i,
                    'Body Part': body_parts.get(i, 'N/A'),
                    'X': landmark.x,
                    'Y': landmark.y,
                    'Z': landmark.z,
                    'Visibility': landmark.visibility
                })

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(landmarks_data)
    if show:
        df.head()

    return results, img_rgb, df

def display_pose_landmarks(image_path):
    pose, mp_drawing, mp_pose = initialize_mediapipe()
    results, img_rgb, df = detect_pose_landmarks(image_path, pose, show=True)

    # Copy image
    img_copy = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Ensure results.pose_landmarks is a list of NormalizedLandmarkList
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Landmarks", img_copy)
        # Wait for a key press and close the window
        cv2.waitKey(0)  # Adjust the wait time for smoother video playback
        cv2.destroyAllWindows()

def initialize_mediapipe ():
    # 2. Load MediaPipe Pose landmark estimation solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    # 3. Load MediaPipe Drawing Utilities
    mp_drawing = mp.solutions.drawing_utils

    return pose, mp_drawing, mp_pose


def plot_pose_landmarks(landmarks_3d, plot_type='3d', show_plot=True, save_path=None):
    """
    Plots pose landmarks in 2D or 3D.

    Parameters:
    - landmarks_3d: numpy array of shape (33, 3) containing pose landmarks.
    - plot_type: '2d' for 2D plotting, '3d' for 3D plotting (default is '3d').
    - show_plot: If True, the plot will be shown (default is True).
    - save_path: If provided, the plot will be saved to this path (default is None).
    """

    # Define connections between landmarks
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
        (0, 9), (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
        (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
        (12, 24), (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
        (23, 24)  # Hips
    ]

    if plot_type == '3d':
        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot landmarks
        ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])

        # Plot connections
        for connection in CONNECTIONS:
            start, end = connection
            ax.plot([landmarks_3d[start, 0], landmarks_3d[end, 0]],
                    [landmarks_3d[start, 1], landmarks_3d[end, 1]],
                    [landmarks_3d[start, 2], landmarks_3d[end, 2]])

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('MediaPipe 3D Pose Landmarks')

        # Adjust the view angle
        ax.view_init(elev=-80, azim=-90)
        #ax.invert_yaxis()

        # Equal scaling for all axis
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    elif plot_type == '2d':
        # Create 2D plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot landmarks in 2D (using X and Y coordinates)
        ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], c='green', zorder=3, s=20, label='Keypoints')

        # Plot connections
        for connection in CONNECTIONS:
            start, end = connection
            ax.plot([landmarks_3d[start, 0], landmarks_3d[end, 0]],
                    [landmarks_3d[start, 1], landmarks_3d[end, 1]], 'b-')

        ax.invert_yaxis()

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('MediaPipe 2D Pose Landmarks')

    else:
        raise ValueError("plot_type must be either '2d' or '3d'.")

    # Show or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    if show_plot:
        plt.show()

    plt.close(fig)  # Close the figure to free memory
