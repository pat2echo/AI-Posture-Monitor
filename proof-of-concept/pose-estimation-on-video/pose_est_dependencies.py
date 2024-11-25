import mediapipe as mp
import cv2
import numpy as np
import pandas as df
import os

from frame_diff_dependencies import FrameDiff
from features_dependencies import  get_features
from predict_dependencies import  predict_pose, get_attr_of_features

class PoseEstimation:
    def process_video(self, video_file=None, label_file=None, scaling_factor=0.5, use_bounding_box=True,
                      model_number=1, is_predict_pose=False, use_frame_diff=True, BASE_OUTPUT_DIR=None):
        self.is_predict_pose = is_predict_pose

        self.model_number = model_number

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        #self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        #self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
        #help(self.pose)
        #return None

        # Open the video file or capture device
        # video_file = "walking_to_sit.mp4"
        cap = cv2.VideoCapture(video_file)

        # Get the FPS of the video
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second: {self.video_fps}")

        # Initialize processing frame interval; set to 0 use recording speed
        processing_interval = 0
        if self.video_fps > 59:
            # limit to 30fps if video is > 30fps
            processing_interval = self.video_fps // 30

        # Get labels
        self.pass_count = 0
        self.fail_count = 0
        self.previous_label = None
        self.label_df = None
        if label_file is not None:
            self.label_df = df.read_csv(label_file)

            self.label_df["action"] = self.label_df["action"].astype(str)
            #label_df["start_time"] = label_df["start_time"].astype(float)
            #label_df["end_time"] = label_df["end_time"].astype(float)

            #label_df["start_frame"] = label_df["start_time"] * self.video_fps
            #label_df["end_frame"] = label_df["end_time"] * self.video_fps
            #print(label_df)

        # Initialize frame variables
        prev_frame = None
        cur_frame = None
        next_frame = None

        # Initialize current frame number
        self.frame_count = -1

        # Initialize interval for saving frames; to ensures not all frames are saved
        self.save_interval = 1

        # Initialize absolute values of frame difference; only frame_diff above this value are processed set to 0 to ignore
        max_abs_threshold = 26
        if not use_bounding_box:
            max_abs_threshold = 30

        # Use merge rectangles after frame differencing
        intersect_rectangles = True

        # Initialize output data for insights
        output_data = []

        self.my_frame_diff = FrameDiff()

        # results folder
        output_results = os.path.join(BASE_OUTPUT_DIR, "output_results")
        os.makedirs(output_results, exist_ok=True)

        # Create a folder to save images
        self.my_frame_diff.output_folder = os.path.join(BASE_OUTPUT_DIR, "output_pose")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder)

        self.my_frame_diff.output_folder2 = os.path.join(BASE_OUTPUT_DIR, "output_pose_o")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder2)

        self.my_frame_diff.output_folder_aoi = os.path.join(BASE_OUTPUT_DIR, "output_aoi")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder_aoi)

        self.my_frame_diff.output_folder_aoi_pose = os.path.join(BASE_OUTPUT_DIR, "output_aoi_pose")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder_aoi_pose)

        # Get font Attributes
        self.get_font_attributes()

        # Sliding window for velocity calculation
        self.window_start_time = 0
        self.average_velocity = 0
        self.window_size = 2
        #self.overlap = int( 0.5 * self.window_size )
        self.overlap = 1
        self.velocity_windows = []

        while True:
            # Get the next frame
            frame, frame_color = self.my_frame_diff.get_frame(cap, scaling_factor=scaling_factor)

            if frame is None:
                break

            if use_frame_diff:
                # Update frame history
                prev_frame = cur_frame
                cur_frame = next_frame
                next_frame = frame

                # Skip until we have 3 frames
                if prev_frame is None or cur_frame is None:
                    continue

            # Increment current frame number
            self.frame_count += 1
            frame_label = 0

            # Set max absolute value to 0 in case frame was not processed
            max_value = 0

            # Initialize control variable to process image or not
            process_image = True

            # Convert frame to RGB for MediaPipe
            #frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
            frame_output = cv2.cvtColor(frame_color.copy(), cv2.COLOR_BGR2RGB)

            features = []
            prediction = None

            if process_image:
                rectangles = None
                area_of_interest = None
                if use_frame_diff:
                    # Perform frame differencing
                    diff_frame = self.my_frame_diff.frame_diff(prev_frame, cur_frame, next_frame, dual_frame_difference=True)

                    if diff_frame is None:
                        process_image = False
                    else:
                        # Get max value of absolute difference
                        max_value = np.max(diff_frame)
                        if max_abs_threshold and max_value < max_abs_threshold:
                            process_image = False

                    if use_bounding_box and process_image:
                        rectangles, area_of_interest = self.my_frame_diff.get_bounding_box(diff_frame=diff_frame, frame_output=frame_output, intersect_rectangles=intersect_rectangles,
                                 frame_count=self.frame_count, save_interval=self.save_interval, show_grid=False, snap_to_grid=True, show_rectangle=True)

                if process_image:
                    predict_rect = None
                    if 'rect' in area_of_interest:
                        predict_rect = [area_of_interest['rect']]
                    elif rectangles is not None and len(rectangles) > 0:
                        predict_rect = rectangles
                    prediction, features, frame_label, _ = self.process_frame(frame_output=frame_output,
                                                                              manual_landmark_drawing=False,
                                                                              use_bounding_box=use_bounding_box,
                                                                              rectangles=predict_rect)

            # Display the result
            frame_output = cv2.cvtColor(frame_output, cv2.COLOR_RGB2BGR)
            cv2.imshow('Motion Detection and Pose Estimation', frame_output)

            # Save frame at regular intervals
            if process_image and self.frame_count % self.save_interval == 0:
                frame_title = f"frame_{self.frame_count:04d}.jpg"
                output_path = os.path.join(self.my_frame_diff.output_folder, frame_title)
                #cv2.imwrite(output_path, frame)
                cv2.imwrite(output_path, frame_output)


                output_path2 = os.path.join(self.my_frame_diff.output_folder2, frame_title)
                cv2.imwrite(output_path2, frame_color)

                # Save max value of absolute difference to csv
                #print(frame_title, max_value, process_image)
                output_data.append([self.frame_count, max_value, process_image, frame_label, prediction, ', '.join(map(str, features))])

            # Check for the ESC key press

            #wait_time = int(1000 / fps)
            wait_time = 1   #fast play
            key = cv2.waitKey(wait_time)
            if key == 27:  # ESC key
                break

        # Save the NumPy array to CSV
        features_attr = get_attr_of_features()
        np.savetxt(os.path.join(output_results, f'{os.path.basename(video_file).split('.')[0]}_results.csv'), np.array(output_data), fmt='%s', delimiter=',',
                   header='file_name,max_value,process_image,label,prediction,' + ','.join(features_attr), comments='')

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    def get_font_attributes(self):
        # Define the text and its position
        self.prefix_text = "Hi Frame"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (0, 255, 0)  # Green color in BGR
        self.font_thickness = 2

    def process_frame(self, frame_output, use_bounding_box=True, rectangles=None, manual_landmark_drawing=False):
        features = []
        prediction = None
        aoi_for_pose = None
        timestamp_secs = 0
        label = None

        if use_bounding_box and rectangles is not None and len(rectangles) > 0:
            # Get area of interest from the biggest frame
            for rect in rectangles:
                (x, y, w, h) = rect
                # aoi_for_pose = frame_output[y:y + h, x:x + w]
                expanded_w = w * 3
                expanded_x_start = max(x - (expanded_w - w) // 2, 0)  # Ensure x doesn't go negative
                expanded_w = min(expanded_w, frame_output.shape[1] - expanded_x_start)  # Ensure width doesn't exceed frame width
                aoi_for_pose = frame_output[y:y + h, expanded_x_start:expanded_x_start + expanded_w]
                break
        elif not use_bounding_box:
            aoi_for_pose = frame_output

        #aoi_for_pose = frame_output.copy()

        if aoi_for_pose is not None and aoi_for_pose.size > 0:
            # aoi_for_pose = cv2.cvtColor(aoi_for_pose, cv2.COLOR_GRAY2RGB)
            results = self.pose.process(aoi_for_pose)

            if results.pose_landmarks:
                # print(results.pose_landmarks)
                if manual_landmark_drawing:
                    self.manual_drwaing_of_landmark(frame=frame_output,
                                                    pose_landmark=results.pose_landmarks,
                                                    pose_connection=self.mp_pose.POSE_CONNECTIONS)
                else:
                    self.mp_drawing.draw_landmarks(
                        frame_output,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 117, 66), thickness=1, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245, 66, 0), thickness=1, circle_radius=2)
                    )

                # Get the frame dimensions and calculate the text position
                timestamp_text, timestamp_secs = self.get_timestamp(frame_count=self.frame_count,
                                                                    video_fps=self.video_fps)

                if self.label_df is not None:
                    timestamp_rounded = np.floor(timestamp_secs)
                    d_label = self.label_df[(self.label_df["start_time"] <= timestamp_rounded) & (self.label_df["end_time"] >= timestamp_rounded)]
                    #print(self.frame_count, d_label, timestamp_rounded)
                    if d_label.shape[0] > 0:
                        label = d_label["action"].values[0].lower()
                        self.previous_label = label
                    else:
                        label = self.previous_label


                frame_height, frame_width = frame_output.shape[:2]

                if self.is_predict_pose:
                    landmarks_data = []
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        landmarks_data.append([landmark.x, landmark.y, landmark.z])

                    all_features = get_features(landmarks_3d=np.array(landmarks_data), image_name=None,
                                            model=self.model_number, return_keypoints=[11, 12, 23, 24, 27, 28])
                    #return_keypoints=[11, 12, 23, 24, 27, 28]: shoulder, hip, ankle
                    features = all_features[0]

                    prediction = predict_pose(features=features)
                    #print(features, prediction)
                    font_color = (255,0,0)
                    pass_fail = 'FAIL'
                    if (label is None and prediction is None) or (label is not None and prediction in label):
                        self.pass_count += 1
                        font_color = (0,0,255)
                        pass_fail = 'PASS'
                    else:
                        self.fail_count += 1


                    # get y-axis of keypoints and calculate velocity
                    keypoints_for_velocity = all_features[1][:,1]

                    self.velocity_windows.append(keypoints_for_velocity)
                    #self.velocity_windows.append(self.frame_count)
                    if len(self.velocity_windows) >= self.window_size + self.overlap:
                        initial_window = self.velocity_windows[:self.window_size]
                        current_window = self.velocity_windows[:self.overlap:]

                        # 4, 5 - ankle
                        # 0, 1 - shoulder
                        # 2, 3 - hip
                        index_pairs = [(4, 0), (5, 1), (4, 2), (5, 3)]

                        initial_window_diff = np.mean(initial_window, axis=0)
                        current_window_diff = np.mean(current_window, axis=0)
                        i_results = [initial_window_diff[second] - initial_window_diff[first] for first, second in index_pairs]
                        c_results = [current_window_diff[second] - current_window_diff[first] for first, second in index_pairs]

                        change = np.array(c_results) - np.array(i_results)
                        self.average_velocity = change[0]
                        #print('initial_window', np.mean(initial_window, axis=0) )
                        #print('current_window', np.mean(current_window, axis=0) )
                        print(f'{self.frame_count:04d} change in y', change )

                        self.velocity_windows = current_window


                    acc = (self.pass_count * 100) / (self.pass_count + self.fail_count)
                    frame_text = f'{pass_fail} - Accuracy: {acc:.2f}, avg. velocity: {self.average_velocity:.2f}'
                    (text_width, text_height), _ = cv2.getTextSize(frame_text, self.font, self.font_scale,
                                                                   self.font_thickness)
                    cv2.putText(frame_output, frame_text, (20, frame_height - text_height - 10), self.font, self.font_scale, font_color,
                                self.font_thickness)

                frame_text = f'Label: {label} Pred: {prediction} {timestamp_text}'
                #frame_text = f'{self.prefix_text} Label: {label} Pred: {prediction} {timestamp_text}'
                (text_width, text_height), _ = cv2.getTextSize(frame_text, self.font, self.font_scale, self.font_thickness)
                text_x = frame_width - text_width - 10  # 10 px padding from the right edge
                text_y = 20  # Position near the top
                cv2.putText(frame_output, frame_text, (text_x, text_y), self.font, self.font_scale, self.font_color,
                            self.font_thickness)

            # Save aoi for pose
            if self.frame_count % self.save_interval == 0:
                self.my_frame_diff.save_image(aoi_for_pose, os.path.join(self.my_frame_diff.output_folder_aoi_pose, f"pose_{self.frame_count:04d}"))
                pass


        return prediction, features, label, timestamp_secs

    def manual_drwaing_of_landmark(self, frame=None, pose_landmark=None, pose_connection=None):
        landmark_coords = {}
        frame_width, frame_height = frame.shape[:2]

        for idx, landmark in enumerate(pose_landmark.landmark):
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            # Boundary check: Ensure the coordinates stay within the image frame
            x = min(max(x, 0), frame_width - 1)
            y = min(max(y, 0), frame_height - 1)

            # Store the coordinates for drawing connections later
            landmark_coords[idx] = (x, y)

            # Draw the landmark point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Add landmark index label
            cv2.putText(frame, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Manually draw the connections using POSE_CONNECTIONS
        if pose_connection is not None:
            for connection in pose_connection:
                start_idx = connection[0]
                end_idx = connection[1]

                # Only draw connections if both landmarks are detected
                if start_idx in landmark_coords and end_idx in landmark_coords:
                    start_point = landmark_coords[start_idx]
                    end_point = landmark_coords[end_idx]

                    # Draw the connection line
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

    def get_timestamp(self, frame_count=None, video_fps=None, export_fps=0):
        # Calculate timestamp in seconds (with fraction for frames_per_second)
        timestamp_sec = frame_count / video_fps

        # Convert timestamp to format hh:mm:ss:ff (including frame fraction)
        hours = int(timestamp_sec // 3600)
        minutes = int((timestamp_sec % 3600) // 60)
        seconds = int(timestamp_sec)
        fraction = 0
        if export_fps > 0:
            fraction = int((timestamp_sec * export_fps) % export_fps)

        timestamp_text = f"frame: {frame_count:04d} - {hours:02}:{minutes:02}:{seconds:02}.{fraction:01} - {timestamp_sec:.6f} - fps: {video_fps:.2f}"
        return timestamp_text, timestamp_sec

    def generate_frames_for_groundtruth(self, video_file=None, frames_per_second=1, BASE_OUTPUT_DIR=None):
        self.my_frame_diff = FrameDiff()
        self.my_frame_diff.output_folder = os.path.join(BASE_OUTPUT_DIR, "output_frames")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder)

        cap = cv2.VideoCapture(video_file)

        # Get video frame rate and set the interval based on frames_per_second
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps // frames_per_second)  # Adjust frame interval to capture specified frames per second

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            # Break if no more frames
            if not ret:
                break

            # Check if the current frame is at the specified interval
            if frame_count % frame_interval == 0:
                timestamp_text, _ = self.get_timestamp(frame_count=frame_count, video_fps=fps, export_fps=frames_per_second)

                # Put timestamp text on frame
                cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                # Save the frame with timestamp, including fraction for unique filename
                frame_filename = os.path.join(self.my_frame_diff.output_folder,
                                              f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

            # Increment frame count
            frame_count += 1

        # Release the video capture object
        cap.release()
        print("Done saving frames.")



