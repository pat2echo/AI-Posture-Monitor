import sys
import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
import shutil

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
                                      connections=pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Landmarks", img_copy)
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def merge_rectangles(rectangles, max_distance=50, pre_padding_percent=1, post_padding_percent=1):
    """
    Merge rectangles that overlap, are within a bigger rectangle, or are in close proximity.

    :param rectangles: List of rectangles in the format (x, y, w, h)
    :param max_distance: Maximum distance between rectangles to be considered for merging
    :param pre_padding_percent: Percentage padding distance to add to rectangles
    :param post_padding_percent: Percentage padding distance to add to rectangles
    :return: List of merged rectangles
    """
    if not rectangles:
        return []

    rectangles = [add_padding(rect, pre_padding_percent) for rect in rectangles]

    rectangles = sort_rectangles(rectangles)

    merged = []

    while len(rectangles) > 0:
        current = rectangles[0]
        rectangles = rectangles[1:]

        i = 0
        while i < len(rectangles):
            rect = rectangles[i]

            # Check for overlap or close proximity
            overlap_x = (current[0] <= rect[0] <= current[0] + current[2] + max_distance) or \
                        (rect[0] <= current[0] <= rect[0] + rect[2] + max_distance)
            overlap_y = (current[1] <= rect[1] <= current[1] + current[3] + max_distance) or \
                        (rect[1] <= current[1] <= rect[1] + rect[3] + max_distance)

            if overlap_x and overlap_y:
                # Merge the rectangles
                current[0] = min(current[0], rect[0])
                current[1] = min(current[1], rect[1])
                current[2] = max(current[0] + current[2], rect[0] + rect[2]) - current[0]
                current[3] = max(current[1] + current[3], rect[1] + rect[3]) - current[1]

                # Remove the merged rectangle
                rectangles = np.delete(rectangles, i, axis=0)
            else:
                i += 1

        # add post merge padding
        current = add_padding(current, post_padding_percent)

        merged.append(current)

    return merged

def sort_rectangles(rectangles):
    # Convert rectangles to a numpy array for easier manipulation
    rectangles = np.array(rectangles)

    # Sort rectangles by area in descending order
    sorted_idx = np.argsort(-(rectangles[:, 2] * rectangles[:, 3]))
    return rectangles[sorted_idx]

def add_padding(rect, padding_percent):
    """
    Add padding to a rectangle.

    :param rect: Rectangle in the format (x, y, w, h)
    :param padding_percent: Padding percentage (0-100)
    :return: Padded rectangle
    """
    pad_x = int(rect[2] * padding_percent / 100)
    pad_y = int(rect[3] * padding_percent / 100)
    return [
        rect[0] - pad_x,
        rect[1] - pad_y,
        rect[2] + 2 * pad_x,
        rect[3] + 2 * pad_y
    ]

def get_area_of_interest(frame, rect):
    """
    Save the portion of the frame within the given rectangle as a new image.

    :param frame: The full frame image
    :param rect: The rectangle coordinates (x, y, w, h)
    """
    x, y, w, h = rect
    return frame[y:y + h, x:x + w]

def save_image(image, image_name):
    """
    Save the portion of the frame within the given rectangle as a new image.

    :param image: Image to save
    :param image_name: Name of the image
    """
    if image is not None and image.size > 0:
        cv2.imwrite(f"{image_name}.jpg", image)

# Function to empty a folder
def empty_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Remove the entire folder
    os.makedirs(folder_path, exist_ok=True)  # Recreate the empty folder


def frame_diff_pose_estimation(video_file=None, scaling_factor=0.5, predict_pose=False, frame_count=-1, BASE_OUTPUT_DIR=None):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open the video file or capture device
    # video_file = "walking_to_sit.mp4"
    cap = cv2.VideoCapture(video_file)

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    # Initialize processing frame interval; set to 0 use recording speed
    processing_interval = 0
    if fps > 59:
        # limit to 30fps if video is > 30fps
        processing_interval = fps // 30

    # Initialize frame variables
    prev_frame = None
    cur_frame = None
    next_frame = None

    # Initialize current frame number
    frame_count = -1

    # Initialize interval for saving frames; to ensures not all frames are saved
    save_interval = 30

    # Initialize absolute values of frame difference; only frame_diff above this value are processed set to 0 to ignore
    max_abs_threshold = 26

    # Initialize intersecting rectangles
    intersect_rectangles = True

    # Initialize output data for insights
    output_data = []

    # Create a folder to save images
    output_folder = os.path.join(BASE_OUTPUT_DIR, "output_pose")
    empty_folder(output_folder)

    output_folder_aoi = os.path.join(BASE_OUTPUT_DIR, "output_aoi")
    empty_folder(output_folder_aoi)

    output_folder_aoi_pose = os.path.join(BASE_OUTPUT_DIR, "output_aoi_pose")
    empty_folder(output_folder_aoi_pose)

    # Define the text and its position
    prefix_text = "Hello Frame"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)  # Green color in BGR
    font_thickness = 2


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

        # Increment current frame number
        frame_count += 1

        # Initialize control variable to process image or not
        process_image = True

        # Set max absolute value to 0 in case frame was not processed
        max_value = 0

        if process_image:
            # Perform frame differencing
            diff_frame = frame_diff(prev_frame, cur_frame, next_frame)

            # Get max value of absolute difference
            max_value = np.max(diff_frame)
            if max_abs_threshold and max_value < max_abs_threshold:
                process_image = False

        # Convert frame to RGB for MediaPipe
        frame_output = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)

        if process_image:
            # Threshold the difference frame
            _, thresh_frame = cv2.threshold(diff_frame, 0.5, 255, cv2.THRESH_BINARY)

            # Find contours in the threshold image
            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get rectangles for all significant contours
            rectangles = []
            for contour in contours:
                if cv2.contourArea(contour) > 800:  # Adjust this threshold as needed
                    rect = cv2.boundingRect(contour)
                    rectangles.append(rect)

            if len(rectangles) > 0:
                if intersect_rectangles:
                    # Merge close or intersecting rectangles
                    rectangles = merge_rectangles(rectangles, max_distance=1000)
                else:
                    # Sort rectangles in desc order by area
                    rectangles = sort_rectangles(rectangles)

                for rect in rectangles:
                    x, y, w, h = rect
                    cv2.rectangle(frame_output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save area of interest (aoi) at regular intervals
                    if frame_count % save_interval == 0:
                        # Get area of interest
                        aoi = get_area_of_interest(frame_output, (x, y, w, h))

                        # Save aoi
                        save_image(aoi, os.path.join(output_folder_aoi, f"object_{frame_count:04d}_{x}_{y}"))

        if process_image and len(rectangles) > 0:
            # Get area of interest from the biggest frame
            aoi_for_pose = None
            for rect in rectangles:
                (x, y, w, h) = rect
                aoi_for_pose = frame_output[y:y + h, x:x + w]
                break

            if aoi_for_pose is not None and aoi_for_pose.size > 0:
                # aoi_for_pose = cv2.cvtColor(aoi_for_pose, cv2.COLOR_GRAY2RGB)
                results = pose.process(aoi_for_pose)

                if results.pose_landmarks:
                    #print(results.pose_landmarks)
                    mp_drawing.draw_landmarks(
                        aoi_for_pose,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 117, 66), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 0), thickness=1, circle_radius=2)
                    )

                    if predict_pose:
                        landmarks_data = []
                        for i, landmark in enumerate(results.pose_landmarks.landmark):
                            landmarks_data.append([landmark.x, landmark.y, landmark.z])

                        #features = get_features(landmarks_3d=np.array(landmarks_data), image_name=None)
                        #print(features, predict_features(features=features))

                        # Get the frame dimensions and calculate the text position
                        frame_label = f'{prefix_text} {frame_count}'
                        frame_height, frame_width = frame_output.shape[:2]
                        (text_width, text_height), _ = cv2.getTextSize(frame_label, font, font_scale, font_thickness)
                        text_x = frame_width - text_width - 10  # 10 px padding from the right edge
                        text_y = 20  # Position near the top
                        cv2.putText(frame_output, frame_label, (text_x, text_y), font, font_scale, font_color, font_thickness)

                # Save aoi for pose
                if frame_count % save_interval == 0:
                    save_image(aoi_for_pose, os.path.join(output_folder_aoi_pose, f"pose_{frame_count:04d}"))
        else:
            # frame_output = frame_rgb
            pass

        # Display the result
        cv2.imshow('Motion Detection and Pose Estimation', frame_output)

        # Save frame at regular intervals
        if frame_count % save_interval == 0:
            frame_title = f"frame_{frame_count:04d}.jpg"
            output_path = os.path.join(output_folder, frame_title)
            cv2.imwrite(output_path, frame_output)

            # Save max value of absolute difference to csv
            print(frame_title, max_value, process_image)
            output_data.append([frame_title, max_value, process_image])

        # Break the loop if 'q' is pressed
        # Check for the ESC key press
        key = cv2.waitKey(1)  # Adjust the wait time for smoother video playback
        if key == 27:  # ESC key
            break

    # Save the NumPy array to CSV
    np.savetxt(os.path.join(output_folder, 'frame_max_values.csv'), np.array(output_data), fmt='%s', delimiter=',',
               header='file_name,max_value,process_image', comments='')

    # Release resources
    cap.release()
    cv2.destroyAllWindows()