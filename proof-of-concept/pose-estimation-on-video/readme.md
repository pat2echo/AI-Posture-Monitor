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
python code/proof-of-concept/pose-estimation-on-video/predict_pose.py ./dataset/hr_fall_detection_3.mp4 1 1 ./code/labels/hr_fall_detection_3.csv
```

Get Ground Truth
```aiignore
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_1.mp4
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_2.mp4
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_3.mp4

python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/fall_detection_4.mp4
```

Analyze Ground Truth
```aiignore
python code/proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./code/labels/hr_fall_detection_3.csv 1

python code/proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./output/output_results/hr_fall_detection_3_results.csv 1 prediction
```

# Color Codes
Yellow: bounding box of the expanded area of interest from memory
Blue: the bounding box by expanding the current area of interest to include the previous area of interest, creating a union of both if they intersect
Green: bounding box around the moving object
Pink: bounding box around the detected pose

Evaluate Static Pose Prediction
```aiignore
python code/proof-of-concept/pose-estimation-on-video/static_pose_eval.py ./output/output_results/_results.csv
```

Track and Plot Velocity of Keypoints
```aiignore
python code/proof-of-concept/pose-estimation-on-video/plot_activity_prediction.py ./output/output_results/hr_fall_detection_1_results.csv
```

Plot Activity Recognition
```aiignore
python code/proof-of-concept/pose-estimation-on-video/transition_plot.py ./output/output_results/hr_fall_detection_3_results.csv
```