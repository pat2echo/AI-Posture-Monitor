import cv2
import sys, os
import mediapipe as mp

from frame_diff_pose_est import get_frame, frame_diff, get_bounding_boxes_connected_components_pad_ratio

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_file = None
scaling_factor = 0.5

if len (sys.argv) > 1:
    print(sys.argv[1])
    video_file = sys.argv[1]
    if len (sys.argv) > 2:
        scaling_factor = float(sys.argv[2])
else:
    print("Expecting 2 arguments: video file and scaling factor")
    exit()

# Open the video file or capture device
#video_file = "walking_to_sit.mp4"
cap = cv2.VideoCapture(video_file)

# Initialize frame variables
prev_frame = None
cur_frame = None
next_frame = None

while True:
    # Get the next frame
    frame = get_frame(cap, scaling_factor=scaling_factor)

    if frame is None:
        break

    # Update frame history
    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = frame

    # Skip until we have 3 frames
    if prev_frame is None or cur_frame is None:
        continue

    # Perform frame differencing
    diff_frame = frame_diff(prev_frame, cur_frame, next_frame)

    # Threshold the difference frame
    _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Get bounding boxes of moving objects
    bounding_boxes = get_bounding_boxes_connected_components_pad_ratio(thresh_frame, frame)

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # Draw bounding boxes and pose landmarks
    frame_output = frame_rgb.copy()
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(frame_output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_output,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 117, 66), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 0), thickness=1, circle_radius=2)
        )

    # Display the result
    cv2.imshow('Motion Detection and Pose Estimation', frame_output)

    # Break the loop if 'q' is pressed
    # Check for the ESC key press
    key = cv2.waitKey(1)  # Adjust the wait time for smoother video playback
    if key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()