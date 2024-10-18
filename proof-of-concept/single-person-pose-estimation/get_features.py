import os
import sys

import numpy as np
import mediapipe as mp
import pandas as pd
from pose_estimation_dependencies import detect_pose_landmarks, get_features

landmarks_3d = None
image_name = None
save_path = None
if len (sys.argv) > 1:
    image_file = sys.argv[1]

    if len(sys.argv) > 2:
        save_path = sys.argv[2]

    # Detect landmarks in image
    image_name = os.path.split(image_file)[1]
    results, img_rgb, df = detect_pose_landmarks(image_path=image_file, pose=None, show=True)
else:
    print("Specify input image file path")

if df is not None:
    landmarks_3d = df[['X','Y','Z']].values

    if save_path is not None:
        os.makedirs(os.path.join("output", save_path), exist_ok=True)  # Recreate the empty folder
        save_path = os.path.join("output", save_path, f'{image_name.split(".")[0]}.csv')

    # Plot landmarks
    get_features(landmarks_3d=landmarks_3d, image_name=image_name.split(".")[0], save_path=save_path)