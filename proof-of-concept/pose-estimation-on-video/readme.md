Plot Histogram
```aiignore
python code/proof-of-concept/pose-estimation-on-video/plot_his.py ./dataset/hr_fall_detection_3.mp4 1
```

Frame Differencing
```aiignore
python code/proof-of-concept/pose-estimation-on-video/frame_diff.py ./dataset/hr_fall_detection_3.mp4 1
```

Predict Pose
```aiignore
python code/proof-of-concept/pose-estimation-on-video/predict_pose.py ./dataset/hr_fall_detection_3.mp4 1 1
```

Get Ground Truth
```aiignore
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_1.mp4
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_2.mp4
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_3.mp4
```

# Color Codes
Yellow: bounding box of the expanded area of interest from memory
Blue: the bounding box by expanding the current area of interest to include the previous area of interest, creating a union of both if they intersect
Green: bounding box around the moving object