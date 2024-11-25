import os.path

import numpy as np
import mediapipe as mp
import cv2
import sys
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pose_estimation_dependencies import detect_pose_landmarks, plot_pose_landmarks

landmarks_3d = None
dimension = '3d'
image_name = None
save_path = None
show_label = False
if len (sys.argv) > 1:
    #print(sys.argv[1])
    image_file = sys.argv[1]

    if len(sys.argv) > 2:
        dimension = sys.argv[2]

        if len(sys.argv) > 3:
            save_path = sys.argv[3]

        if len(sys.argv) > 4 and int(sys.argv[4]) > 0:
            show_label = True

    # Detect landmarks in image
    image_name = os.path.split(image_file)[1]
    results, img_rgb, df = detect_pose_landmarks(image_path=image_file, pose=None, show=True)
else:
    print("Specify input image file path")

if df is not None:
    print(df[['X','Y','Z']].values.shape)
    #print(df.head())
    print(df[df.index.isin([11, 12, 23, 24, 27, 28])])
    #print(df[['X', 'Y', 'Z']].head())
    landmarks_3d = df[['X','Y','Z']].values

    if save_path is not None:
        os.makedirs(os.path.join("output", save_path), exist_ok=True)  # Recreate the empty folder
        save_path = os.path.join("output", save_path, f'{dimension}_{image_name}')

    # Plot landmarks
    plot_pose_landmarks(landmarks_3d=landmarks_3d, plot_type=dimension, show_plot=True, save_path=save_path, has_labels=show_label)