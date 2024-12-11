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
python code/proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./code/labels/fall_detection_4.csv 1

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

## Define Fall
When the subject remains in the fallen state for at least 1 second excluding transition to the fallen state

## How to Prepare New Dataset
1. Extract keyframes from the video at a rate of one frame per second to represent the temporal evolution of the activity
```aiignore
python code/proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/fall_detection_4.mp4
```
2. Create a label csv file by recording the activities for each second, shown below is an example csv file
```aiignore
start_time,end_time,action,is_fall
0,10,None,False
11,32,Stand,False
33,41,Stand,False
42,91,Stand,False
92,93,Stand,False
94,95,Stand-Lie,True
96,98,Lie,True
99,100,Lie-Stand,False
101,111,Stand,False
112,112,Stand-Sit,False
113,115,Sit,False
116,116,Sit-Lie,False
116,116,Sit-Lie,False
```
3. Visually validate the labelled data by verifying that the plot follows a logical pattern
```aiignore
python code/proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./code/labels/fall_detection_4.csv 0
```
**Note:** the second argument can accept values of 1 or 0
-- 1: Show all classes in the plot
-- 0: Compress to only key classes (Stand, Sit, Lie)
  
This plot will give you an idea of the class balance  

4. Fall Detection
Use the video and labels to detect and validate falls
`python code/proof-of-concept/pose-estimation-on-video/predict_pose.py VIDEO_FILE MAKE_PREDICTION SCALING_FACTOR LABEL_CSV_FILE`
**Note:**
-- VIDEO_FILE: file path to the video file
-- MAKE_PREDICTION: accepts 1 or 0 
-- SCALING_FACTOR: accepts >= 0.1 
-- LABEL_CSV_FILE: file path to the csv label file 

Example:
```aiignore
python code/proof-of-concept/pose-estimation-on-video/predict_pose.py ./dataset/fall_detection_4.mp4 1 1 ./code/labels/fall_detection_4.csv
```
