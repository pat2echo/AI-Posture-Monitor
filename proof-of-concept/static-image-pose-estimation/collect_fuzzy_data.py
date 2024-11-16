import os
import sys

import pandas as pd

from pose_estimation_dependencies import detect_pose_boundingbox, initialize_mediapipe

import time

landmarks_3d = None
image_name = None
save_path = None
if len(sys.argv) > 1:
    image_file = sys.argv[1]

    if os.path.isfile(image_file):
        # Detect bounding box aspect ratio and relative height/width in a single image
        image_name = os.path.split(image_file)[1]
        dic = detect_pose_boundingbox(image_path=image_file, pose=None, show=True)
        print(dic)
    elif os.path.isdir(image_file):
        # Initialize mediapipe
        pose, mp_drawing, mp_pose = initialize_mediapipe()

        all_bbox_data = {}

        # Get all jpg or png files in the directory
        for file in os.listdir(image_file):
            if file.endswith(('.jpg', '.png')):
                image_name = os.path.split(file)[1]
                full_path = os.path.join(image_file, file)

                print('image_file', image_name)
                dic = detect_pose_boundingbox(image_path=full_path, pose=pose, show=False)

                # If all_bbox_data is empty, initialize it with keys from dic and empty lists as values
                if not all_bbox_data:
                    all_bbox_data['image_name'] = []
                    for key in dic:
                        all_bbox_data[key] = []

                # Append values from the detected data (dic) to the corresponding keys in all_bbox_data
                all_bbox_data['image_name'].append(image_name)
                for key, value in dic.items():
                    all_bbox_data[key].append(value)

                time.sleep(2)
            else:
                print(f"Skipping non-image file: {file}")

        # Merge features_field_names and all_features, convert to DataFrame and save to CSV
        if all_bbox_data:
            df_features = pd.DataFrame(all_bbox_data)
            save_path = os.path.join(image_file, 'static_pose_boundingbox_data.csv')
            df_features.to_csv(save_path, index=False)
            print(f"Features saved to {save_path}")
        else:
            print("No valid image files found in the directory.")
else:
    print("Specify input image file path or directory path with jpg or png files.")