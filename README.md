# ai_posture_monitor: Real-Time AI-Powered Posture Monitoring

This project, `ai_posture_monitor`, is designed to be an innovative and cost-effective solution for real-time activity monitoring of elderly individuals. It utilizes the MediaPipe pose estimation model, along with fuzzy logic and finite state machines, to achieve reliable tracking, posture recognition, and fall detection.

## Contents
- Key Functionalities
- Try the Demo (Kaggle or CLI)
- How it Works
- Using Source Codes in Repo
- Other Utility Scripts
- Color Codes
- Dataset
- Self-Consent Form for Dataset Usage


## Key Functionalities

* **Real-time Activity Monitoring:** Continuously tracks and analyzes movements, providing a comprehensive view of activity patterns.
* **Pose Detection:** Accurately identifies key postures such as standing, sitting, and lying down.
* **Fall Detection:** Effectively detects falls with minimal false alarms, offering peace of mind.
* **Fuzzy Logic Analysis:** Employs fuzzy logic for accurate interpretation of movement data, enhancing the system's reliability.
* **User-Centric Design:** Specifically designed for the needs of elderly individuals living alone, providing a user-friendly experience.
* **Environmental Adaptability:** Functions effectively in well-lit indoor settings, suitable for typical home environments.
* **Scalable and Cost-Effective:** Represents an affordable solution with potential for diverse applications in elderly care.

## Try the Demo (Kaggle or CLI)

**Supported environments:** Kaggle notebooks and local CLI/Jupyter with Python **3.9 - 3.12**
(mediapipe has no 3.13+ wheels in the supported range). Google Colab is not officially
supported — its runtime preloads libraries that conflict with mediapipe's pinned dependencies.

### Option 1 — Kaggle, zero setup (easiest)
Open the published demo notebook and click **Copy & Edit**, then **Run All**:

&nbsp;&nbsp;&nbsp;&nbsp;**https://www.kaggle.com/code/patrickogbuitepu/fall-detection-posture-classification-starter**

It runs the package on sample images from the
[dataset](https://www.kaggle.com/datasets/patrickogbuitepu/posture-monitor-and-fall-detection),
explores the activity/fall labels, and visualizes the classifier's precomputed per-frame output.
The same notebook lives in this repo at
[notebooks/fall_detection_posture_classification_demo.ipynb](notebooks/fall_detection_posture_classification_demo.ipynb).

### Option 2 — CLI on your machine
Clone the repo, set up a virtual environment, install, and run:
```bash
# Linux / macOS / WSL (use python3.12/3.11/3.10 - not 3.13+):
git clone https://github.com/pat2echo/AI-Posture-Monitor.git
cd AI-Posture-Monitor
python3.12 -m venv .venv
source .venv/bin/activate
pip install ai-posture-monitor kagglehub
python examples/cli_demo.py
```
```powershell
# Windows (PowerShell or cmd):
git clone https://github.com/pat2echo/AI-Posture-Monitor.git
cd AI-Posture-Monitor
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install ai-posture-monitor kagglehub
python examples\cli_demo.py
```
This fetches 3 sample images (stand/sit/lie) from the published dataset (public - no Kaggle
account needed), runs pose estimation + the rule-based posture classifier, prints a verdict
table, and saves landmark-annotated images to `./output`. Expected output:
```
image                        stand                    sit                      lie
--------------------------------------------------------------------------------------------------
stand.jpg                    standing/standing        non_sitting/non_sitting  non_lying/non_lying
sit.jpg                      non_standing/non_standing sitting/sitting          non_lying/non_lying
lie.jpg                      no pose landmarks detected

Annotated images saved to .../output
```
(`lie.jpg` reporting no landmarks is expected - MediaPipe struggles with some lying poses,
which is why the full system also uses bounding-box features as a fallback.)

Run on your own photos with `python examples/cli_demo.py --images your1.jpg your2.jpg`,
and see `python examples/cli_demo.py --help` for all options.

## How it Works
1. Install the package  
   `pip install ai-posture-monitor`  

2. Predict Static Posture on your video file
```
import ai_posture_monitor as pm

pe = pm.PoseEstimation()
pe.process_video(video_file=video_file, plot_results=True, predict_fall=False)
```

3. Fall Detection on your video file
```
import ai_posture_monitor as pm

pe = pm.PoseEstimation()
pe.process_video(video_file=video_file, plot_results=True)
```

----

## Using Source Codes in Repo
Alternatively, you can use the project source code to run a complete workflow from validating your labels to predictions and outputting your results  

Install the dependencies:
```aiignore
pip install numpy
pip install opencv-python
pip install "mediapipe>=0.10.14,<0.10.22"
pip install pandas
pip install scikit-learn
pip install matplotlib
```
**Note:** mediapipe must stay below 0.10.22 — the legacy `mp.solutions` API this
project is built on was removed in mediapipe 0.10.30+.
Pre-requisite: this prototype is built to be executed from the command line only

### Definition of Fall
When the subject remains in the fallen state for at least 1 second excluding transition to the fallen state

### How to Prepare New Dataset
1. Extract keyframes from the video at a rate of one frame per second to represent the temporal evolution of the activity
```aiignore
python proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/fall_detection_4.mp4
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
python proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./labels/fall_detection_4.csv 0
```
**Note:** the second argument can accept values of 1 or 0  
-- 1: Show all classes in the plot  
-- 0: Compress to only key classes (Stand, Sit, Lie)  
  
This plot will give you an idea of the class balance  

4. Fall Detection
Use the video and labels to detect and validate falls  
`python proof-of-concept/pose-estimation-on-video/predict_fall.py VIDEO_FILE MAKE_PREDICTION SCALING_FACTOR LABEL_CSV_FILE`  
**Note:**  
-- VIDEO_FILE: file path to the video file  
-- MAKE_PREDICTION: accepts 1 or 0   
-- SCALING_FACTOR: accepts >= 0.1   
-- LABEL_CSV_FILE: file path to the csv label file   

You can also perform posture classification only with  
`python proof-of-concept/pose-estimation-on-video/predict_pose.py VIDEO_FILE MAKE_PREDICTION SCALING_FACTOR LABEL_CSV_FILE`  

Example:
```aiignore
python proof-of-concept/pose-estimation-on-video/predict_fall.py ./dataset/fall_detection_9.mp4 1 1 ./labels/fall_detection_9.csv
python proof-of-concept/pose-estimation-on-video/predict_pose.py ./dataset/fall_detection_9.mp4 1 1 ./labels/fall_detection_9.csv
```

5. Visualize Fall Plot
```aiignore
python proof-of-concept/pose-estimation-on-video/fall_plot.py ./output/output_results/fall_detection_4_results.csv
```

---

## Other Utility Scripts
Plot Histogram
```aiignore
python proof-of-concept/pose-estimation-on-video/plot_his.py ./dataset/hr_fall_detection_3.mp4 1
```

Frame Differencing
```aiignore
python proof-of-concept/pose-estimation-on-video/frame_diff.py ./dataset/hr_fall_detection_3.mp4 1
```

Predict Pose
```aiignore
python proof-of-concept/pose-estimation-on-video/predict_pose.py ./dataset/hr_fall_detection_3.mp4 1 1
python proof-of-concept/pose-estimation-on-video/predict_pose.py ./dataset/hr_fall_detection_3.mp4 1 1 ./labels/hr_fall_detection_3.csv
```

Get Ground Truth
```aiignore
python proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_1.mp4
python proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_2.mp4
python proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/hr_fall_detection_3.mp4

python proof-of-concept/pose-estimation-on-video/groundtruth.py ./dataset/fall_detection_4.mp4
```

Analyze Ground Truth
```aiignore
python proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./labels/hr_fall_detection_3.csv 1
python proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./labels/fall_detection_4.csv 1

python proof-of-concept/pose-estimation-on-video/analyze_manual_label.py ./output/output_results/hr_fall_detection_3_results.csv 1 prediction
```

# Color Codes
Yellow: bounding box of the expanded area of interest from memory  
Blue: the bounding box by expanding the current area of interest to include the previous area of interest, creating a union of both if they intersect  
Green: bounding box around the moving object  
Pink: bounding box around the detected pose  
  
Evaluate Static Pose Prediction  
```aiignore
python proof-of-concept/pose-estimation-on-video/static_pose_eval.py ./output/output_results/_results.csv
```

Track and Plot Velocity of Keypoints
```aiignore
python proof-of-concept/pose-estimation-on-video/plot_activity_prediction.py ./output/output_results/hr_fall_detection_1_results.csv
```

Plot Activity Recognition
```aiignore
python proof-of-concept/pose-estimation-on-video/transition_plot.py ./output/output_results/hr_fall_detection_3_results.csv
```


## Dataset: Video Files used in this Experiment
The full dataset (10 videos, 113 static pose images, labels, and derived feature CSVs) is
published on Kaggle under CC BY 4.0:
**https://www.kaggle.com/datasets/patrickogbuitepu/posture-monitor-and-fall-detection**

Original archival copy (University of Essex OneDrive, may require institutional access):
https://essexuniversity-my.sharepoint.com/:f:/g/personal/po23102_essex_ac_uk/End63nA718NNjDdOPOjRaMABtli7MI-JnkAZwXesGFe2KA?e=Bmxtjf

---

# Self-Consent Form for Dataset Usage

> **Note:** this form is reproduced as signed in August 2024, when the dataset was
> restricted to University of Essex storage. Explicit approval for public release was
> subsequently obtained, and the dataset is now published on Kaggle under CC BY 4.0
> (see the Dataset section above).

**Project Title:** AI-Driven Posture Analysis Fall Detection System for the Elderly

**Researcher Name:** Patrick O. Ogbuitepu

**Purpose of the Dataset:**

This dataset, consisting of 113 static images and 9 video recordings, will be used solely for academic research and development of an AI-powered fall detection system for the elderly as part of my dissertation project.

**Dataset Details:**

* **Static Images:** 113 self-portraits capturing various static poses.
* **Videos:** 9 recordings of myself performing daily activities relevant to the research.
* **Storage Location:** University of Essex OneDrive
* **Access Link:** https://essexuniversity-my.sharepoint.com/:f:/g/personal/po23102_essex_ac_uk/End63nA718NNjDdOPOjRaMABtli7MI-JnkAZwXesGFe2KA?e=Bmxtjf

**Consent:**

By signing below, I acknowledge the following:

* I am the sole subject in the dataset and willingly recorded the images and videos for the stated purpose.
* I consent to the use of this dataset in my dissertation project, including analysis, algorithm development, and result validation.
* I understand the dataset will be stored securely on the University of Essex OneDrive and will not be shared publicly without explicit approval.
* I retain the right to withdraw consent for dataset usage at any time, acknowledging this may affect the research project's progress. 
* I understand and comply with the University of Essex's ethical guidelines for using personal data in research.

**Signature:** Patrick O. Ogbuitepu

**Date:** 31-Aug-2024

