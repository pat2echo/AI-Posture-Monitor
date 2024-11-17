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
python code/proof-of-concept/static-image-pose-estimation/plot_pose.py dataset/pose/sit.jpg 3d second_pose

# plot, save and show label in plot
python code/proof-of-concept/static-image-pose-estimation/plot_pose.py dataset/pose/sit.jpg 3d second_pose/label 1
```

Get Features from Keypoints
```aiignore
python code/proof-of-concept/static-image-pose-estimation/get_features.py dataset/pose/sit.jpg
```

Get All Features
```aiignore
python code/proof-of-concept/static-image-pose-estimation/get_features.py ./dataset/pose
```


Predict All Features
```aiignore
python code/proof-of-concept/static-image-pose-estimation/predict.py ./dataset/pose/features_output.csv

python code/proof-of-concept/static-image-pose-estimation/predict.py ./dataset/pose/features_output.csv relative_joints
python code/proof-of-concept/static-image-pose-estimation/predict.py ./dataset/pose/static_pose_boundingbox_data.csv bounding_box_size
```

Get Evaluation Metrics
```aiignore
python code/proof-of-concept/static-image-pose-estimation/metrics.py ./dataset/pose/features_output_predicted.csv

# Run all
python code/proof-of-concept/static-image-pose-estimation/get_features.py ./dataset/pose && python code/proof-of-concept/static-image-pose-estimation/predict.py ./dataset/pose/features_output.csv && python code/proof-of-concept/static-image-pose-estimation/metrics.py ./dataset/pose/features_output_predicted.csv
python code/proof-of-concept/static-image-pose-estimation/get_features.py ./dataset/video-pose-2 && python code/proof-of-concept/static-image-pose-estimation/predict.py ./dataset/pose/features_output.csv && python code/proof-of-concept/static-image-pose-estimation/metrics.py ./dataset/pose/features_output_predicted.csv
```


Get Aspect Ratio, and Relative Width / Height for Fuzzy Set Thresholds
```aiignore
python code/proof-of-concept/static-image-pose-estimation/collect_fuzzy_data.py ./dataset/pose/sit.jpg

python code/proof-of-concept/static-image-pose-estimation/collect_fuzzy_data.py ./dataset/pose
```

Analysis to get thresholds for fuzzy sets
```aiignore
python code/proof-of-concept/static-image-pose-estimation/clustering.py ./dataset/pose/static_pose_boundingbox_data.csv kmeans aspect_ratio 3

python code/proof-of-concept/static-image-pose-estimation/clustering.py ./dataset/pose/static_pose_boundingbox_data.csv histogram aspect_ratio 30
```