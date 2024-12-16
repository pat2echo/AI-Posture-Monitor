# ai_posture_monitor: Real-Time AI-Powered Posture Monitoring

This project, `ai_posture_monitor`, is designed to be an innovative and cost-effective solution for real-time activity monitoring of elderly individuals. It utilizes the MediaPipe pose estimation model, along with fuzzy logic and finite state machines, to achieve reliable tracking, posture recognition, and fall detection.

## Contents
- Key Functionalities
- How it Works
- Other Utility Scripts
- Color Codes
- Evaluation
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
pip install mediapipe
pip install pandas
pip install scikit-learn
pip install matplotlib
```
Pre-requisite: this prototype is built to be executed from the command line only

### Definition of Fall
When the subject remains in the fallen state for at least 1 second excluding transition to the fallen state

```aiignore
python ./tests/predict_falls.py 0 1 1 0 1
python ./tests/predict_video_static_pose.py .py 0 1 1 0 1
```