###########
# Resources
###########
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
#
# Pose Landmarks
#   https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
#   https://ai.google.dev/static/edge/mediapipe/images/solutions/pose_landmarks_index.png
#
# AI Hand Pose Estimation with MediaPipe and Python
#   https://youtu.be/vQZ4IvB07ec?si=OFZ5LE-7qWyL-LZv
#   https://github.com/nicknochnack/MediaPipeHandPose/blob/main/Handpose%20Tutorial.ipynb
#
# Real-Time 3D Pose Detection & Pose Classification | Mediapipe | OpenCV | Python
#   https://youtu.be/aySurynUNAw?si=2K-0Gk89XXWuB_ZL
#
###########
# Implementation
###########
# 1. install in terminal and load libraries
# pip install mediapipe opencv-python
import sys

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pandas as pd

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

def detect_pose_landmarks(image_path, pose, show=False):
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

def display_pose_landmarks(image_path, pose, mp_drawing):
    results, img_rgb, df = detect_pose_landmarks(image_path, pose, show=True)

    # Copy image
    img_copy = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Ensure results.pose_landmarks is a list of NormalizedLandmarkList
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Landmarks", img_copy)
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 2. Load MediaPipe Pose landmark estimation solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# 3. Load MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# 4. Get Input Image
if len (sys.argv) > 1:
    print(sys.argv[1])
    image_file = sys.argv[1]

    # 5. Detect landmarks in image
    #image_file = "woman_sitting.png"
    display_pose_landmarks(image_file, pose, mp_drawing)
else:
    print("Specify input image file path")
