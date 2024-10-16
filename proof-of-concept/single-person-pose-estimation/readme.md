# Notes

Installation
```
pip install mediapipe opencv-python
```

Import Libraries
```angular2html
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
```

Pose Estimation with MediaPipe
```angular2html
python.exe .\get_pose_landmark.py .\woman_sitting.png

# Arg 1: Image File
```

Plot Pose
```aiignore
python code/proof-of-concept/single-person-pose-estimation/plot_pose.py dataset/pose/sit.jpg 3d second_pose
```