# Notes

Installation
```
pip install mediapipe opencv-python
```

Import Libraries
```
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
```


Sample Pose Estimation with MediaPipe
```
python.exe .\frame_pose.py M:\work\project\hr_fall_detection_3.mp4
python code/proof-of-concept/pose-estimation/frame_pose.py ./dataset/hr_fall_detection_3.mp4 

python.exe .\frame_pose.py .\walking_to_sit.mp4 0.3

```

## Scenarios
- Applied max threshold of absolute values of frame difference
- Merged multiple intersecting & close proximity bounding boxes into 1
- Applied padding before merging and after merging


## Results
- https://youtu.be/bu6oqsvODlI

- https://youtu.be/2a2-LL6jxLE