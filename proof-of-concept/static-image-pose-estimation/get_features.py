import os
import sys

import numpy as np
import mediapipe as mp
import pandas as pd

from pose_estimation_dependencies import detect_pose_landmarks, get_features, get_attr_of_features, predict_features

import time

landmarks_3d = None
image_name = None
save_path = None
if len(sys.argv) > 1:
    image_file = sys.argv[1]

    if os.path.isfile(image_file):
        # Detect landmarks in a single image
        image_name = os.path.split(image_file)[1]
        results, img_rgb, df = detect_pose_landmarks(image_path=image_file, pose=None, show=True)

        if df is not None and not df.empty:
            landmarks_3d = df[['X', 'Y', 'Z']].values
            features = get_features(landmarks_3d=landmarks_3d, image_name=image_name.split(".")[0])
            print(features, predict_features(features=features), df.describe() )
        else:
            print(f"No landmarks detected in image: {image_name}")

    elif os.path.isdir(image_file):
        features_field_names = get_attr_of_features()
        all_features = []

        # Get all jpg or png files in the directory
        for file in os.listdir(image_file):
            if file.endswith(('.jpg', '.png')):
                image_name = os.path.split(file)[1]
                full_path = os.path.join(image_file, file)

                results, img_rgb, df = detect_pose_landmarks(image_path=full_path, pose=None, show=True)
                if df is not None and not df.empty:
                    landmarks_3d = df[['X', 'Y', 'Z']].values
                    features = get_features(landmarks_3d=landmarks_3d, image_name=image_name.split(".")[0])
                    all_features.append(features)
                else:
                    # Create an empty list for this image if no landmarks are detected
                    features = [image_name] + [None] * (len(features_field_names) - 1)
                    all_features.append(features)

                time.sleep(2)
            else:
                print(f"Skipping non-image file: {file}")

        # Merge features_field_names and all_features, convert to DataFrame and save to CSV
        if all_features:
            df_features = pd.DataFrame(all_features, columns=features_field_names)
            save_path = os.path.join(image_file, 'features_output.csv')
            df_features.to_csv(save_path, index=False)
            print(f"Features saved to {save_path}")
        else:
            print("No valid image files found in the directory.")
else:
    print("Specify input image file path or directory path with jpg or png files.")