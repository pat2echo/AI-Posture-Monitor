import sys
import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd


# Function to capture and resize frames
def get_frame(cap, scaling_factor=None, res=None):
    def get_frame_scale_down(cap, scaling_factor):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def get_frame_resized(cap, res):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if scaling_factor is not None:
        return get_frame_scale_down(cap, scaling_factor)
    else:
        return get_frame_resized(cap, res)


# Function to calculate frame difference
def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)


# Function to get bounding boxes using connected components with padding ratio
def get_bounding_boxes_connected_components_pad_ratio(thresh_frame, original_frame, padding_ratio=0.1):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_frame)
    height, width = original_frame.shape[:2]

    bounding_boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:
            padding_w = int(w * padding_ratio)
            padding_h = int(h * padding_ratio)
            x_padded = max(0, x - padding_w)
            y_padded = max(0, y - padding_h)
            w_padded = min(width - x_padded, w + 2 * padding_w)
            h_padded = min(height - y_padded, h + 2 * padding_h)
            bounding_boxes.append((x_padded, y_padded, w_padded, h_padded))

    return bounding_boxes


def get_rgb_image_from_cv2(image_path, show=False):
    # Read image
    img = cv2.imread(image_path)
    img_rgb = None

    # Check if the image was loaded successfully
    if img is None:
        print("Error: Image not loaded. Check the file path.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if show:
            # Display the image in a window named 'Image'
            cv2.imshow(image_path.split('.')[0], img)

            # Wait for a key press and close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return img_rgb

def pose_landmarks():
    # Dictionary mapping landmark indices to body parts
    landmark_dict = {
        0: 'nose',
        1: 'left eye (inner)',
        2: 'left eye',
        3: 'left eye (outer)',
        4: 'right eye (inner)',
        5: 'right eye',
        6: 'right eye (outer)',
        7: 'left ear',
        8: 'right ear',
        9: 'mouth (left)',
        10: 'mouth (right)',
        11: 'left shoulder',
        12: 'right shoulder',
        13: 'left elbow',
        14: 'right elbow',
        15: 'left wrist',
        16: 'right wrist',
        17: 'left pinky',
        18: 'right pinky',
        19: 'left index',
        20: 'right index',
        21: 'left thumb',
        22: 'right thumb',
        23: 'left hip',
        24: 'right hip',
        25: 'left knee',
        26: 'right knee',
        27: 'left ankle',
        28: 'right ankle',
        29: 'left heel',
        30: 'right heel',
        31: 'left foot index',
        32: 'right foot index'
    }

    return landmark_dict

def detect_pose_landmarks(image_path, pose, show=False):
    img_rgb = get_rgb_image_from_cv2(image_path, show=False)
    results = pose.process(img_rgb)

    landmarks_data = []
    if results.pose_landmarks:
        if show:
            # Extract landmark data
            body_parts = pose_landmarks()
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append({
                    'Landmark': i,
                    'Body Part': body_parts.get(i, 'N/A'),
                    'X': landmark.x,
                    'Y': landmark.y,
                    'Z': landmark.z,
                    'Visibility': landmark.visibility
                })

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(landmarks_data)
    if show:
        df.head()

    return results, img_rgb, df

def display_pose_landmarks(image_path, pose, mp_drawing):
    results, img_rgb, df = detect_pose_landmarks(image_path, pose, show=True)

    # Copy image
    img_copy = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Ensure results.pose_landmarks is a list of NormalizedLandmarkList
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Landmarks", img_copy)
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()