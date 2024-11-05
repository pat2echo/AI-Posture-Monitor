import mediapipe as mp
import cv2
import numpy as np
import os
from frame_diff_dependencies import FrameDiff
from features_dependencies import  get_features
from predict_dependencies import  predict_pose, get_attr_of_features

class PoseEstimation:
    def process_video(self, video_file=None, scaling_factor=0.5, use_bounding_box=True,
                      model_number=1, is_predict_pose=False, use_frame_diff=True, BASE_OUTPUT_DIR=None):
        self.is_predict_pose = is_predict_pose

        self.model_number = model_number

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #help(self.pose)
        #return None

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

        # Create a folder to save images
        self.my_frame_diff.output_folder = os.path.join(BASE_OUTPUT_DIR, "output_pose")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder)

        self.my_frame_diff.output_folder_aoi = os.path.join(BASE_OUTPUT_DIR, "output_aoi")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder_aoi)

        self.my_frame_diff.output_folder_aoi_pose = os.path.join(BASE_OUTPUT_DIR, "output_aoi_pose")
        self.my_frame_diff.empty_folder(self.my_frame_diff.output_folder_aoi_pose)

        # Get font Attributes
        self.get_font_attributes()


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
                                 frame_count=self.frame_count, save_interval=self.save_interval, show_grid=False, snap_to_grid=True)

                if process_image:
                    predict_rect = None
                    if 'rect' in area_of_interest:
                        predict_rect = [area_of_interest['rect']]
                    elif rectangles is not None and len(rectangles) > 0:
                        predict_rect = rectangles
                    prediction, features = self.process_frame(frame_output=frame_output, use_bounding_box=use_bounding_box, rectangles=predict_rect)

            # Display the result
            frame_output = cv2.cvtColor(frame_output, cv2.COLOR_RGB2BGR)
            cv2.imshow('Motion Detection and Pose Estimation', frame_output)

            # Save frame at regular intervals
            if process_image and self.frame_count % self.save_interval == 0:
                frame_title = f"frame_{self.frame_count:04d}.jpg"
                output_path = os.path.join(self.my_frame_diff.output_folder, frame_title)
                #cv2.imwrite(output_path, frame)
                cv2.imwrite(output_path, frame_output)

                # Save max value of absolute difference to csv
                print(frame_title, max_value, process_image)
                output_data.append([frame_title, max_value, process_image, prediction, ', '.join(map(str, features))])

            # Check for the ESC key press

            #wait_time = int(1000 / fps)
            wait_time = 1   #fast play
            key = cv2.waitKey(wait_time)
            if key == 27:  # ESC key
                break

        # Save the NumPy array to CSV
        features_attr = get_attr_of_features()
        np.savetxt(os.path.join(self.my_frame_diff.output_folder, 'frame_max_values.csv'), np.array(output_data), fmt='%s', delimiter=',',
                   header='file_name,max_value,process_image,prediction,' + ','.join(features_attr), comments='')

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    def get_font_attributes(self):
        # Define the text and its position
        self.prefix_text = "Hello Frame"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (0, 255, 0)  # Green color in BGR
        self.font_thickness = 2

    def process_frame(self, frame_output, use_bounding_box=True, rectangles=None):
        features = []
        prediction = None
        aoi_for_pose = None

        if use_bounding_box and rectangles is not None and len(rectangles) > 0:
            # Get area of interest from the biggest frame
            for rect in rectangles:
                (x, y, w, h) = rect
                aoi_for_pose = frame_output[y:y + h, x:x + w]
                break
        elif not use_bounding_box:
            aoi_for_pose = frame_output

        if aoi_for_pose is not None and aoi_for_pose.size > 0:
            # aoi_for_pose = cv2.cvtColor(aoi_for_pose, cv2.COLOR_GRAY2RGB)
            results = self.pose.process(aoi_for_pose)

            if results.pose_landmarks:
                # print(results.pose_landmarks)
                self.mp_drawing.draw_landmarks(
                    aoi_for_pose,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 117, 66), thickness=1, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 0), thickness=1, circle_radius=2)
                )

                if self.is_predict_pose:
                    landmarks_data = []
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        landmarks_data.append([landmark.x, landmark.y, landmark.z])

                    features = get_features(landmarks_3d=np.array(landmarks_data), image_name=None, model=self.model_number)
                    prediction = predict_pose( features=features)
                    print(features, prediction)

                    # Get the frame dimensions and calculate the text position
                    frame_label = f'{self.prefix_text} {prediction} {self.frame_count}'
                    frame_height, frame_width = frame_output.shape[:2]
                    (text_width, text_height), _ = cv2.getTextSize(frame_label, self.font, self.font_scale, self.font_thickness)
                    text_x = frame_width - text_width - 10  # 10 px padding from the right edge
                    text_y = 20  # Position near the top
                    cv2.putText(frame_output, frame_label, (text_x, text_y), self.font, self.font_scale, self.font_color,
                                self.font_thickness)

            # Save aoi for pose
            if self.frame_count % self.save_interval == 0:
                self.my_frame_diff.save_image(aoi_for_pose, os.path.join(self.my_frame_diff.output_folder_aoi_pose, f"pose_{self.frame_count:04d}"))
                pass


        return prediction, features

    def generate_frames_for_groundtruth(self, video_file=None):
        cap = cv2.VideoCapture(video_file)

        # Get video frame rate and set frame interval to 1 frame per second
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Number of frames to skip to get 1 frame per second

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            # Break if no more frames
            if not ret:
                break

            # Check if the current frame is at the 1-second interval
            if frame_count % frame_interval == 0:
                # Calculate timestamp in seconds
                timestamp_sec = frame_count // int(fps)

                # Convert timestamp to format hh:mm:ss
                timestamp_text = f"{timestamp_sec // 3600:02}:{(timestamp_sec % 3600) // 60:02}:{timestamp_sec % 60:02}"

                # Put timestamp text on frame
                cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                # Save the frame with timestamp
                frame_filename = os.path.join(self.my_frame_diff.output_folder, f"frame_{timestamp_sec:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

            # Increment frame count
            frame_count += 1

        # Release the video capture object
        cap.release()
        print("Done saving frames.")

