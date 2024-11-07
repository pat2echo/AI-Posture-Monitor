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
from pose_estimation_dependencies import *

# 4. Get Input Image
if len (sys.argv) > 1:
    print(sys.argv[1])
    image_file = sys.argv[1]

    # 5. Detect landmarks in image
    #image_file = "woman_sitting.png"
    display_pose_landmarks(image_file)
else:
    print("Specify input image file path")
