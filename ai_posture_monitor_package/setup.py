from setuptools import setup, find_packages

setup(
    name='ai_posture_monitor',
    version='0.0.22',
    # mediapipe ships no wheels for Python 3.13+ in the supported range; without
    # this bound, pip on 3.13 backtracks to ancient broken releases of this package
    python_requires='>=3.9,<3.13',
    packages=find_packages(),
    description='A package designed to predict static pose and detect falls with 2D RGB Camera in well lit indoor environments.',
    long_description="""# AI Posture Monitor
This project introduces an innovative, cost-effective solution for real-time activity monitoring of elderly individuals. By leveraging the MediaPipe pose estimation model, fuzzy logic, and finite state machines, the system can reliably track individuals, recognize static postures (standing, sitting, lying), and detect transitions, particularly focusing on falls.

**Source code:** https://github.com/pat2echo/AI-Posture-Monitor  
**Dataset (CC BY 4.0):** https://www.kaggle.com/datasets/patrickogbuitepu/posture-monitor-and-fall-detection  
**Demo notebook on Kaggle:** https://www.kaggle.com/code/patrickogbuitepu/fall-detection-posture-classification-starter  

## Requirements
Python **3.9 - 3.12** (mediapipe, the pose-estimation dependency, ships no wheels for 3.13+
in the supported version range).

Supported environments: local CLI/Jupyter and Kaggle notebooks. Google Colab is not
officially supported (its runtime preloads libraries that conflict with mediapipe's
pinned dependencies).

## Quickstart
Set up a virtual environment and install:
```bash
# Linux / macOS / WSL (use python3.12/3.11/3.10 if your default python3 is 3.13+):
python3.12 -m venv .venv && source .venv/bin/activate
pip install ai-posture-monitor kagglehub
```
```
# Windows:
py -3.12 -m venv .venv && .venv\\Scripts\\activate
pip install ai-posture-monitor kagglehub
```

Classify a static pose in a few lines (fetches a sample image from the public
Kaggle dataset - no Kaggle account needed):
```python
import kagglehub
import ai_posture_monitor as pm

img = kagglehub.dataset_download(
    'patrickogbuitepu/posture-monitor-and-fall-detection', path='train/pose/stand.jpg')

pose, mp_drawing, mp_pose = pm.initialize_mediapipe()
results, img_rgb, landmarks = pm.detect_pose_landmarks(img, pose=pose, show=True)
print(pm.get_features(landmarks[['X', 'Y', 'Z']].to_numpy(), image_name='stand.jpg'))
# -> ['stand.jpg', True, 59, 'standing', 'standing', 82, 86, 'non_sitting', ...]
```

## CLI demo
A ready-made command-line demo (prints a stand/sit/lie verdict table and saves
landmark-annotated images) lives in the repository:
```bash
git clone https://github.com/pat2echo/AI-Posture-Monitor
python AI-Posture-Monitor/examples/cli_demo.py                  # 3 sample images from the dataset
python AI-Posture-Monitor/examples/cli_demo.py --images my.jpg  # your own photos
```

## Try it on Kaggle (zero setup)
Open the demo notebook and click "Copy & Edit":
https://www.kaggle.com/code/patrickogbuitepu/fall-detection-posture-classification-starter

## Fall detection on video
```python
import ai_posture_monitor as pm

pe = pm.PoseEstimation()
pe.process_video(video_file='my_video.mp4', plot_results=True)                      # fall detection
pe.process_video(video_file='my_video.mp4', plot_results=True, predict_fall=False)  # posture only
```

## Features
- **Real-Time Activity Monitoring**: Tracks and analyzes movements continuously.
- **Pose Detection**: Identifies key postures: standing, sitting, and lying down.
- **Fall Detection**: Detects falls with minimal false alarms.
- **Fuzzy Logic Analysis**: Utilizes fuzzy logic for accurate movement interpretation.
- **User-Centric Design**: Tailored for elderly individuals living alone.
- **Environmental Adaptability**: Functions effectively in well-lit indoor settings.
- **Scalable and Cost-Effective**: Affordable solution for diverse applications.
""",
    long_description_content_type='text/markdown',
    author='Patrick Ogbuitepu',
    author_email='pat2echo@gmail.com',
    keywords=['Fall Detection', 'Human Motion Classification', 'Pose Estimation', 'Elderly Care', 'Computer Vision', 'Machine Learning', 'Real-time Monitoring', 'Occlusion Handling', 'Finite State Machine', 'Activity Recognition', 'Elderly Care Technology', 'Pose Estimation Models', 'Real-Time Motion Tracking', 'Occlusion Handling', 'Static Pose Classification', 'Fuzzy Logic Systems', 'Temporal Pose Analysis', 'Human Pose Detection', 'Motion Detection Algorithms', 'Inactivity Monitoring', 'Rehabilitation Monitoring', 'Bounding Box Analysis', 'Multi-Camera Systems', 'Pose Transition Detection', 'Computational Intelligence Applications', 'Human-Centric AI', 'Assistive Technologies', 'Vision-Based Health Monitoring', 'Sensorless Activity Recognition', 'Human-Computer Interaction (HCI)', 'Machine Learning in Elderly Care', 'AI for Healthcare Monitoring', 'Zero False Alarm Systems', 'Dim Light Pose Recognition', 'Occlusion Robustness', 'Home Monitoring Systems'],
    install_requires=[
        'numpy',
        'opencv-python',
        # >=0.10.14: last release without a numpy<2 pin; <0.10.22: the legacy
        # mp.solutions API this package is built on was removed in 0.10.30+
        'mediapipe>=0.10.14,<0.10.22',
        'pandas',
        'scikit-learn',
        'matplotlib'
    ]
)