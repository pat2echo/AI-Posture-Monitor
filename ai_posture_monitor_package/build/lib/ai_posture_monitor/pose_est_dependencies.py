import mediapipe as mp
import cv2
import numpy as np
import pandas as df
import os

from .frame_diff_dependencies import FrameDiff
from .features_dependencies import  get_features
from .predict_dependencies import  predict_pose, get_attr_of_features
from .plot_dependencies import plot_fall_data, plot_label_vs_prediction_data, DynamicPlot

class PoseEstimation:
    def process_video(self, video_file=None, label_file=None, scaling_factor=0.8, use_bounding_box=True,
                      model_number=2, is_predict_pose=True, use_frame_diff=True, BASE_OUTPUT_DIR=None, debug_mode=True, plot_results=False, predict_fall=True):
        self.is_predict_pose = is_predict_pose

        self.model_number = model_number
        self.debug_mode = debug_mode

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
        if video_file is None:
            print("Video file was not specified, using webcam")
            cap = cv2.VideoCapture(0)
        else:
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
        if BASE_OUTPUT_DIR is not None:
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

            self.save_data = True
            self.my_frame_diff.save_data = True
        else:
            self.my_frame_diff.save_data = False
            self.save_data = False

        # Get font Attributes
        self.get_font_attributes()

        #keypoints_for_velocity=[11, 12, 23, 24, 27, 28]: shoulder, hip, ankle
        keypoints_for_velocity = [11, 12, 23, 24]

        # Sliding window for velocity calculation
        self.window_start_time = 0
        self.max_velocity = 0
        self.max_acceleration = 0
        self.window_size = 16
        #self.overlap = int( 0.5 * self.window_size )
        self.overlap = 8
        self.velocity_windows = []
        self.tmp_data = []
        self.acceleration_windows = []
        self.velocity_sequence = []

        features_attr = get_attr_of_features()

        # Memory used to smooth predictions and ignore anomalies
        self.prediction_memory = []
        self.prediction_memory_length = 5
        self.prediction_memory_velocity = []
        self.prediction_memory_length_velocity = 16
        self.previous_prediction = None
        self.previous_prediction_bbox = None
        self.use_bbox_prediction_model = False

        # Monitor transitions over 40 frames
        self.state_transition_window = 40
        self.state_transition_memory = []

        # Memory used to track aspect ratio moving average
        self.aspect_ratio_window = 4
        self.aspect_ratio_memory = []
        self.aspect_ratio_lie_threshold_short = 1.5
        self.aspect_ratio_lie_threshold_long = 1.2

        # Fall
        self.previous_label_fall = None
        self.state_transition_sequence = []
        self.state_transition_window = 45
        self.fall_label_sequence = []
        self.fall_label_window = 30
        self.fall_watch_prediction = False
        self.fall_watch = False
        self.fall_count = 0
        self.fall_pass = 0
        self.fall_fail = 0
        self.previous_fall_prediction_states = {}
        self.fall_alert_prediction_sequence = []
        self.fall_alert_prediction_sequence_window = 9
        self.fall_alert_sequence = []
        self.fall_alert = False

        empty_features = np.full(len(features_attr), np.nan)

        # dynamic plot
        showDynamicPlot = False if self.debug_mode else True
        dynamicPlot = None
        if showDynamicPlot:
            dynamicPlot = DynamicPlot()


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
            #if self.frame_count < 1000:
            #    continue
            frame_label = 0

            # Set max absolute value to 0 in case frame was not processed
            max_value = 0

            # Initialize control variable to process image or not
            process_image = True

            # Convert frame to RGB for MediaPipe
            #frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
            frame_output = cv2.cvtColor(frame_color.copy(), cv2.COLOR_BGR2RGB)

            features = empty_features
            prediction = None
            initial_prediction = None
            aspect_ratio_prediction = None
            state_transition_prediction = None
            bbox_aspect_ratio_of_pose = 0
            velocity = np.full(len(keypoints_for_velocity), np.nan)
            acceleration = velocity
            aoi_from_memory = False
            bbox_aspect_ratio = 0
            is_motion = False
            is_transition = False
            aspect_ratio_movement = 0
            state_sequence = None

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

                    is_motion = process_image
                    if use_bounding_box and process_image:
                        rectangles, area_of_interest = self.my_frame_diff.get_bounding_box(diff_frame=diff_frame, frame_output=frame_output, intersect_rectangles=intersect_rectangles,
                                 frame_count=self.frame_count, save_interval=self.save_interval, show_grid=False, snap_to_grid=True, show_rectangle=self.debug_mode)

                if process_image:
                    if 'memory' in area_of_interest:
                        aoi_from_memory = area_of_interest['memory']

                    if 'bbox_aspect_ratio' in area_of_interest:
                        bbox_aspect_ratio = area_of_interest['bbox_aspect_ratio']

                    predict_rect = None
                    if 'rect' in area_of_interest:
                        predict_rect = [area_of_interest['rect']]
                    elif rectangles is not None and len(rectangles) > 0:
                        predict_rect = rectangles
                    is_transition, initial_prediction, prediction, state_transition_prediction, state_sequence, aspect_ratio_prediction, bbox_aspect_ratio_of_pose, aspect_ratio_movement, features, frame_label, _, velocity = self.process_frame(frame_output=frame_output,
                                                                                                      manual_landmark_drawing=False,
                                                                                                      bbox_aspect_ratio=bbox_aspect_ratio,
                                                                                                      keypoints_of_focus=keypoints_for_velocity,
                                                                                                      use_bounding_box=use_bounding_box, predict_fall=predict_fall,
                                                                                                      bounding_boxes=predict_rect)
                    self.update_fall_prediction_sequence(state=self.fall_watch_prediction)
                else:
                    self.update_tranistion_sequence(state=None)
                    self.update_fall_prediction_sequence(state=None)

            # Display the result
            frame_output = cv2.cvtColor(frame_output, cv2.COLOR_RGB2BGR)
            frame_title = 'Posture Classification Mode (ESC Key to Close)'
            if predict_fall:
                frame_title = 'Fall Detection Mode (ESC Key to Close)'

            if showDynamicPlot:
                plot_pred = prediction
                if predict_fall:
                    plot_pred = self.fall_alert
                frame_output = dynamicPlot.plot(prediction=plot_pred, frame_output=frame_output, offset=self.frame_count, type='fall' if predict_fall else 'pose')

            cv2.imshow(frame_title, frame_output)

            # Save frame at regular intervals
            if process_image and self.frame_count % self.save_interval == 0:
                if self.save_data:
                    frame_title = f"frame_{self.frame_count:04d}.jpg"
                    output_path = os.path.join(self.my_frame_diff.output_folder, frame_title)
                    #cv2.imwrite(output_path, frame)
                    cv2.imwrite(output_path, frame_output)

                    output_path2 = os.path.join(self.my_frame_diff.output_folder2, frame_title)
                    cv2.imwrite(output_path2, frame_color)

                # Save max value of absolute difference to csv
                #print(frame_title, max_value, process_image)
                #output_data.append([self.frame_count, frame_label, initial_prediction, prediction, state_transition_prediction, state_sequence, self.fall_watch, self.fall_watch_prediction, aspect_ratio_prediction, is_transition, aoi_from_memory, bbox_aspect_ratio_of_pose, aspect_ratio_movement, bbox_aspect_ratio, max_value, is_motion, ', '.join(map(str, list(velocity))), ', '.join(map(str, features)) ])
                if len(features) == 0:
                    features = empty_features
                out_one = [self.frame_count, frame_label, initial_prediction, prediction, state_transition_prediction, state_sequence, self.fall_alert, self.fall_watch, self.fall_watch_prediction, aspect_ratio_prediction, is_transition, aoi_from_memory, bbox_aspect_ratio_of_pose, aspect_ratio_movement, bbox_aspect_ratio, max_value, is_motion] + list(velocity) + list(features)
                output_data.append(out_one)

            # Check for the ESC key press

            #wait_time = int(1000 / fps)
            wait_time = 1   #fast play
            key = cv2.waitKey(wait_time)
            if key == 27:  # ESC key
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        if showDynamicPlot:
            dynamicPlot.close()

        # Save the NumPy array to CSV
        plot_title = 'webcam'
        if video_file is not None:
            plot_title = os.path.basename(video_file)

        if plot_results:
            df_for_plot = df.DataFrame(output_data, columns=['file_name', 'label', 'prediction', 'smooth_prediction',
                                                    'state_transition_prediction', 'state_sequence', 'fall', 'fall_watch',
                                                    'fall_watch_prediction', 'aspect_ratio_prediction', 'is_transition',
                                                    'aoi_from_memory', 'bbox_aspect_ratio_of_pose',
                                                    'aspect_ratio_movement', 'bbox_aspect_ratio', 'max_value',
                                                    'is_motion'] + ['v_lshoulder', 'v_rshoulder', 'v_lhip',
                                                                    'v_rhip'] + features_attr)


            if predict_fall:
                plot_fall_data(df=df_for_plot, plot_title=f'fall_{plot_title}')
            else:
                plot_label_vs_prediction_data(df=df_for_plot, plot_title=f'static_pose_{plot_title}', plot_size=(10,8))

        if self.save_data:
            np.savetxt(os.path.join(output_results, f'{plot_title.split('.')[0]}_results.csv'), np.array(output_data), fmt='%s', delimiter=',',
                   header='file_name,label,prediction,smooth_prediction,state_transition_prediction,state_sequence,fall,fall_watch,fall_watch_prediction,aspect_ratio_prediction,is_transition,aoi_from_memory,bbox_aspect_ratio_of_pose,aspect_ratio_movement,bbox_aspect_ratio,max_value,is_motion,' + ','.join(['v_lshoulder','v_rshoulder','v_lhip','v_rhip']) +','+ ','.join(features_attr), comments='')


    def get_font_attributes(self):
        # Define the text and its position
        self.prefix_text = "Hi Frame"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (0, 255, 0)  # Green color in BGR
        self.font_thickness = 2

    def denoise_prediction(self, initial_prediction=None, aspect_ratio_prediction=None, bbox_aspect_ratio_of_pose=0, aspect_ratio_movement=0):
        # Keep track of last consecutive predictions and only accept new state when it persists over a number of frames
        self.prediction_memory.append(initial_prediction)
        if len(self.prediction_memory) >= (self.prediction_memory_length + 1):
            self.prediction_memory.pop(0)

        has_prediction_changed = False
        state_has_transitioned = False

        array = np.array(self.prediction_memory)
        if np.all(array == array[0]):
            # test if both shoulders velocity has changed
            state_has_transitioned = self.detect_transition_from_prev_state(prev_state=self.previous_prediction, current_state=initial_prediction)
            smooth_prediction = initial_prediction
            self.previous_prediction = initial_prediction
            has_prediction_changed = True
        else:
            smooth_prediction = self.previous_prediction

        return smooth_prediction, has_prediction_changed, state_has_transitioned

    def adjustment_for_sit_in_activity_recognition(self, initial_prediction=None, smooth_prediction=None, aspect_ratio_prediction=None, bbox_aspect_ratio_of_pose=0, aspect_ratio_movement=0):
        # Use bounding box prediction to improve lie prediction when sit is predicted
        if self.use_bbox_prediction_model:
            if aspect_ratio_prediction != 'lie' or aspect_ratio_prediction != self.previous_prediction_bbox or initial_prediction == aspect_ratio_prediction:
                self.use_bbox_prediction_model = False
                smooth_prediction = initial_prediction
            else:
                smooth_prediction = aspect_ratio_prediction
                self.previous_prediction_bbox = smooth_prediction
        elif smooth_prediction == 'sit' and aspect_ratio_prediction == 'lie' and bbox_aspect_ratio_of_pose >= aspect_ratio_movement:
            smooth_prediction = aspect_ratio_prediction
            self.use_bbox_prediction_model = True
            self.previous_prediction_bbox = smooth_prediction
        elif smooth_prediction == 'sit' and initial_prediction == 'lie' and aspect_ratio_prediction == 'sit':
            smooth_prediction = initial_prediction
            self.previous_prediction_bbox = smooth_prediction

        return smooth_prediction

    def detect_transition(self, initial_prediction=None, velocity=[], has_prediction_changed=False, aspect_ratio_prediction=None, bbox_aspect_ratio=0):
        # Keep track of last consecutive joint downward velocity
        self.prediction_memory_velocity.append(velocity)
        if len(self.prediction_memory_velocity) >= (self.prediction_memory_length_velocity + 1):
            self.prediction_memory_velocity.pop(0)

        is_transition = False
        if has_prediction_changed:
            # test if both shoulders velocity has changed
            arr = np.array(self.prediction_memory_velocity)
            if len( arr[(arr[:, 0] != 0) & (arr[:, 1] != 0)] ) > 0 or self.previous_prediction is None:
                is_transition = True
            elif len(arr[(arr[:, 0] != 0) | (arr[:, 1] != 0)]) > 0 and ( (initial_prediction == aspect_ratio_prediction) or (initial_prediction != self.previous_prediction != aspect_ratio_prediction) ):
                    is_transition = True

        return is_transition

    def detect_transition_from_prev_state(self, prev_state=None, current_state=None):
        transition = False
        if '-' in current_state:
            # test for transition label
            transition = True
        elif prev_state is not None and current_state != prev_state and (
                '-' not in prev_state or '-' not in current_state):
            # test for transition label
            transition = True

        return transition

    def process_frame(self, frame_output, use_bounding_box=True, bounding_boxes=None, keypoints_of_focus=None, bbox_aspect_ratio=0, manual_landmark_drawing=False, predict_fall=True):
        features = []
        prediction = None
        initial_prediction = None
        aspect_ratio_prediction = None
        state_transition_prediction = None
        bbox_aspect_ratio_of_pose = 0
        aoi_for_pose = None
        timestamp_secs = 0
        label = None
        label_fall = None
        is_transition = False
        aspect_ratio_movement = 0
        state_sequence = None
        has_no_fall_prediction = True

        #keypoints_of_focus=[11, 12, 23, 24, 27, 28]: shoulder, hip, ankle

        velocity = np.zeros_like(keypoints_of_focus)
        acceleration = np.zeros_like(keypoints_of_focus)

        if use_bounding_box and bounding_boxes is not None and len(bounding_boxes) > 0:
            # Get area of interest from the biggest frame
            for rect in bounding_boxes:
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
                if not self.debug_mode:
                    # skip drawing pose
                    pass
                elif manual_landmark_drawing:
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
                                                                    video_fps=self.video_fps, show_full_info=self.debug_mode)

                if self.label_df is not None:
                    timestamp_rounded = np.floor(timestamp_secs)
                    d_label = self.label_df[(self.label_df["start_time"] <= timestamp_rounded) & (self.label_df["end_time"] >= timestamp_rounded)]
                    #print(self.frame_count, d_label, timestamp_rounded)
                    if d_label.shape[0] > 0:
                        label = d_label["action"].values[0].lower()
                        self.previous_label = label
                        label_fall = d_label["is_fall"].values[0]
                        self.previous_label_fall = label_fall
                    else:
                        label = self.previous_label
                        label_fall = self.previous_label_fall


                frame_height, frame_width = frame_output.shape[:2]

                # update fall label sequence
                self.fall_label_sequence.append(label_fall)
                if len(self.fall_label_sequence) >= self.fall_label_window:
                    self.fall_label_sequence.pop(0)

                fall_non_fall = '-'
                fall_font_color = (255, 0, 0)

                arr = np.array(self.fall_label_sequence)
                if arr[arr == True].size > 0:
                    # watch fall window
                    self.fall_watch = True

                if self.fall_watch:
                    fall_non_fall = 'Fall'

                if self.is_predict_pose:
                    landmarks_data = []
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        landmarks_data.append([landmark.x, landmark.y, landmark.z])

                    all_features = get_features(landmarks_3d=np.array(landmarks_data), image_name=None,
                                                model=self.model_number, return_keypoints=keypoints_of_focus)
                    features = all_features[0]

                    # get aspect ratio of bounding box surrounding the landmark
                    bbox_aspect_ratio_of_pose = self.get_bounding_box_of_detected_pose(frame=frame_output, pose_landmark=np.array(landmarks_data), show=self.debug_mode)

                    # get y-axis of keypoints and calculate velocity
                    keypoints_for_velocity = all_features[1][:,1]

                    self.velocity_windows.append(keypoints_for_velocity)
                    #self.tmp_data.append([self.frame_count] + list(keypoints_for_velocity))
                    #self.velocity_windows.append(self.frame_count)

                    if len(self.velocity_windows) >= self.window_size + self.overlap:
                        initial_window = self.velocity_windows[:self.window_size]
                        current_window = self.velocity_windows[self.overlap:]

                        initial_window_diff = np.max(initial_window, axis=0)
                        current_window_diff = np.max(current_window, axis=0)

                        velocity = np.array(current_window_diff) - np.array(initial_window_diff)

                        # smooth velocity ensuring sequence of last 3 values has not crossed zero
                        self.velocity_sequence.append(velocity)
                        if len(self.velocity_sequence) > 3:
                            self.velocity_sequence.pop(0)
                        if len(self.velocity_sequence) == 3:
                            #print('self.velocity_sequence')
                            #print(self.velocity_sequence)
                            # Check each column
                            columns = np.array(self.velocity_sequence).T
                            results = [
                                True if np.all(col > 0) else True if np.all(col < 0) else False
                                for i, col in enumerate(columns)
                            ]

                            # Print results
                            #print('smooth velocity', results)

                            if results.count(True) >= 2:
                                valid = np.zeros_like(velocity)
                                for i, v in enumerate(results):
                                    valid[i] = velocity[i] if v else 0
                                velocity = valid
                                #print('smooth velocity', velocity)
                            else:
                                velocity = np.zeros_like(velocity)
                        else:
                            velocity = np.zeros_like(velocity)

                        self.max_velocity = np.max(velocity)
                        #print('initial_window', np.mean(initial_window, axis=0) )
                        #print('current_window', np.mean(current_window, axis=0) )
                        #print(f'{self.frame_count:04d} velocity in y-axis', velocity )

                        self.acceleration_windows.append(velocity)
                        if len(self.acceleration_windows) >= (self.window_size + self.overlap) // 2:
                            a_initial_window = self.acceleration_windows[:(self.window_size//2)]
                            a_current_window = self.acceleration_windows[(self.overlap//2):]

                            a_initial_window_diff = np.max(a_initial_window, axis=0)
                            a_current_window_diff = np.max(a_current_window, axis=0)

                            acceleration = np.array(a_current_window_diff) - np.array(a_initial_window_diff)
                            self.max_acceleration = np.max(acceleration)

                            self.acceleration_windows = a_current_window

                        self.velocity_windows = current_window


                    initial_prediction = predict_pose(features=features)
                    aspect_ratio_prediction = self.predict_posture_from_aspect_ratio(aspect_ratio=bbox_aspect_ratio_of_pose)
                    aspect_ratio_movement = self.track_aspect_ratio_movement(aspect_ratio=bbox_aspect_ratio_of_pose)
                    prediction, has_changed, is_transition = self.denoise_prediction(initial_prediction=initial_prediction, bbox_aspect_ratio_of_pose=bbox_aspect_ratio_of_pose, aspect_ratio_movement=aspect_ratio_movement, aspect_ratio_prediction=aspect_ratio_prediction)
                    state_transition_prediction = self.adjustment_for_sit_in_activity_recognition(initial_prediction=initial_prediction, smooth_prediction=prediction, aspect_ratio_prediction=aspect_ratio_prediction, bbox_aspect_ratio_of_pose=bbox_aspect_ratio_of_pose, aspect_ratio_movement=aspect_ratio_movement)
                    #is_transition = self.detect_transition(initial_prediction=initial_prediction, has_prediction_changed=has_changed, bbox_aspect_ratio=bbox_aspect_ratio, velocity=velocity, aspect_ratio_prediction=aspect_ratio_prediction)

                    self.update_tranistion_sequence(state=state_transition_prediction)
                    #print('sequence', self.state_transition_sequence)

                    fall_prediction = self.get_fall_prediction(np.array(self.state_transition_sequence))
                    state_sequence = fall_prediction['seq']

                    has_no_fall_prediction = False
                    is_fallen = False
                    if 'fall' in fall_prediction and fall_prediction['fall'] > 0:
                        is_fallen = True
                        fall_non_fall = f'{fall_non_fall}+Fall'
                        self.fall_watch_prediction = True
                        fall_font_color = (0,0,255)

                    #if self.previous_fall_prediction_states is None or (self.previous_fall_prediction_states is not None and 'seq' in self.previous_fall_prediction_states and self.previous_fall_prediction_states['seq'] != state_sequence):
                    self.predict_fall_alert(state_sequence=fall_prediction)

                    self.previous_fall_prediction_states = fall_prediction

                    if self.fall_watch and arr[arr == True].size == 0:
                        self.fall_count += 1
                        if self.fall_watch_prediction:
                            self.fall_pass += 1
                        else:
                            self.fall_fail += 1
                        # End of fall watch
                        self.fall_watch = False

                    if not is_fallen:
                        self.fall_watch_prediction = False

                    #print(features, prediction)
                    font_color = (255,0,0)
                    pass_fail = 'FAIL'
                    if (label is None and prediction is None) or (label is not None and prediction is not None and prediction in label):
                        self.pass_count += 1
                        font_color = (0,0,255)
                        pass_fail = 'PASS'
                    else:
                        self.fail_count += 1


                    prev_text_height = 0
                    if self.label_df is not None:
                        acc = (self.pass_count * 100) / (self.pass_count + self.fail_count)
                        frame_text = f'Pose: {pass_fail} - Accuracy: {acc:.2f}, aspect ratio: {bbox_aspect_ratio_of_pose:.3f}/{aspect_ratio_movement}, max velocity: {self.max_velocity:.3f}, trans: {state_transition_prediction}'
                        (text_width, text_height), _ = cv2.getTextSize(frame_text, self.font, self.font_scale, self.font_thickness)
                        prev_text_height = text_height
                        cv2.putText(frame_output, frame_text, (20, frame_height - text_height - 10), self.font, self.font_scale, font_color, self.font_thickness)

                        if predict_fall:
                            acc = (self.fall_pass * 100) / (self.fall_count) if self.fall_count > 0 else 0
                            frame_text = f'{fall_non_fall} - Accuracy: {acc}, Label: {label_fall} {str(fall_prediction)}'
                            (text_width, text_height), _ = cv2.getTextSize(frame_text, self.font, self.font_scale,
                                                                           self.font_thickness)
                            cv2.putText(frame_output, frame_text, (20, frame_height - (text_height + prev_text_height + 20) ), self.font, self.font_scale, fall_font_color, self.font_thickness)

                else:
                    self.update_tranistion_sequence(state=None)


                state_text = 'Transition ' if is_transition else ''


                if not self.debug_mode:
                    self.font_scale = 1.3
                    self.font_thickness = 4
                    text_y = 50
                else:
                    text_y = 20

                if self.label_df is None:
                    frame_text = f'Pose: {prediction.upper()} {timestamp_text}'
                    if predict_fall:
                        frame_text = f'Fall: {1 if self.fall_alert else 0} {frame_text}'
                else:
                    frame_text = f'{state_text}Label: {label} Pred: {prediction} {timestamp_text}'

                #frame_text = f'{self.prefix_text} Label: {label} Pred: {prediction} {timestamp_text}'
                (text_width, text_height), _ = cv2.getTextSize(frame_text, self.font, self.font_scale, self.font_thickness)
                text_x = frame_width - text_width - 10  # 10 px padding from the right edge
                cv2.putText(frame_output, frame_text, (text_x, text_y), self.font, self.font_scale, self.font_color,
                            self.font_thickness)

            # Save aoi for pose
            if self.frame_count % self.save_interval == 0:
                if self.save_data:
                    self.my_frame_diff.save_image(aoi_for_pose, os.path.join(self.my_frame_diff.output_folder_aoi_pose, f"pose_{self.frame_count:04d}"))

        # reset fall pred in live mode
        if has_no_fall_prediction and self.label_df is None:
            self.fall_watch_prediction = None

        return is_transition, initial_prediction, prediction, state_transition_prediction, state_sequence, aspect_ratio_prediction, bbox_aspect_ratio_of_pose, aspect_ratio_movement, features, label, timestamp_secs, velocity

    def get_finite_states(self):
        # State transitions and fall/no fall
        return {
            'stand-sit': {'fall': 0, 'end_episode': 0},
            'stand-sit-stand': {'fall': 0, 'end_episode': 2},
            #'stand-sit-stand': {'fall': 0, 'end_episode': 1},
            'stand-sit-lie': {'fall': 2, 'end_episode': 1},
            'sit-stand': {'fall': 0, 'end_episode': 2},
            #'sit-stand': {'fall': 0, 'end_episode': 1},
            'stand-lie': {'fall': 2, 'end_episode': 0},
            'stand-lie-stand': {'fall': 0, 'end_episode': 2},
            #'stand-lie-stand': {'fall': 1, 'end_episode': 1},
            'lie-stand': {'fall': 0, 'end_episode': 2},
            #'lie-stand': {'fall': 0, 'end_episode': 1},
            'sit-lie': {'fall': 0, 'end_episode': 2},
            #'sit-lie': {'fall': 0, 'end_episode': 0},
            'sit-lie-sit': {'fall': 0, 'end_episode': 0},
            'lie-sit': {'fall': 0, 'end_episode': 2},
            'lie-sit-lie': {'fall': 2, 'end_episode': 1},
            'lie-lie': {'fall': 1, 'end_episode': 0},  # leave out of scope, check for downward velocity
        }

    def update_tranistion_sequence(self, state):
        self.state_transition_sequence.append(state)
        if len(self.state_transition_sequence) >= self.state_transition_window:
            self.state_transition_sequence.pop(0)

    def update_fall_prediction_sequence(self, state=None):
        self.fall_alert_prediction_sequence.append(state)
        if len(self.fall_alert_prediction_sequence) > self.fall_alert_prediction_sequence_window:
            self.fall_alert_prediction_sequence.pop(0)

    def predict_fall_alert(self, state_sequence=None):
        #if self.fall_alert_prediction_sequence[-1] == True
        true_count = np.sum(np.array(self.fall_alert_prediction_sequence) == True)  # `== True` filters for True only
        list_length = len(self.fall_alert_prediction_sequence)
        #print('predfall 1', self.frame_count, true_count / list_length, self.fall_alert_prediction_sequence)
        #print(state_sequence)
        #self.fall_alert = True if true_count / list_length > 0.5 else False
        self.fall_alert_sequence.append(True if true_count / list_length > 0.5 else False)
        # retain values for only 1 second
        if len(self.fall_alert_sequence) > 30:
            self.fall_alert_sequence.pop(0)

        #print(state_sequence)
        if state_sequence is not None and 'end_episode' in state_sequence:
            if state_sequence['end_episode'] > 1:
                self.fall_alert_sequence = []
                self.fall_alert = False
            else:
                true_count = np.sum(np.array(self.fall_alert_sequence) == True)
                list_length = len(self.fall_alert_sequence)
                #print('predfall 2', self.frame_count, true_count / list_length, self.fall_alert_sequence)

                self.fall_alert = True if true_count / list_length > 0.2 else False
        elif len(self.fall_alert_sequence) >= 30:
            true_count = np.sum(np.array(self.fall_alert_sequence) == True)
            list_length = len(self.fall_alert_sequence)
            #print('predfall 3', self.frame_count, true_count / list_length, self.fall_alert_sequence)
            self.fall_alert = True if true_count / list_length > 0.8 else False

    def first_non_none(self, data_array):
        for i, x in enumerate(data_array):
            if x is not None:
                return x, i
        return None, -1

    def get_fall_prediction(self, data_array):
        data_array = np.array(data_array)
        dict_prediction = {'seq':None}
        if len(data_array) > 1:

            first = self.first_non_none(data_array)
            first_state = data_array[data_array == first[0]]
            current_state = None
            t_state = data_array[(data_array != None)]
            if len(t_state) > 1:
                current_state = t_state[-1]

           #if self.frame_count > 1330:
           #    print(self.frame_count, data_array)

            mid_state = data_array[(data_array != first[0]) & (data_array != None)]
            # Exclude mid-state if less than 20% of non None window values
            # print(data_array[data_array != mid_state[0]])
            #print('current_state', current_state)
            #print(first, '\n',first_state, len(first_state),'\n', mid_state, len(mid_state))

            size = len(data_array) - 1
            percent_dist = {}
            percent_dist[first_state[0]] = np.round(len(first_state) * 100 / size, 0)

            sequence = None
            if len(mid_state) > 0:
                if first_state[0] == mid_state[0] or current_state == mid_state[0]:
                    sequence = f'{first_state[0]}-{current_state}'
                else:
                    sequence = f'{first_state[0]}-{mid_state[0]}-{current_state}'
                    percent_dist[mid_state[0]] = np.round(len(mid_state) * 100 / size, 0)
            elif current_state is not None:
                sequence = f'{first_state[0]}-{current_state}'

            finite_states = self.get_finite_states()
            if sequence in finite_states:
                dict_prediction = finite_states[sequence]
            dict_prediction['seq'] = sequence
            dict_prediction['per'] = percent_dist

        return dict_prediction

    def track_aspect_ratio_movement(self, aspect_ratio=0):
        self.aspect_ratio_lie_threshold_short = 1.5
        self.aspect_ratio_lie_threshold_long = 1.2

        self.aspect_ratio_memory.append(aspect_ratio)
        if len(self.aspect_ratio_memory) >= ((self.aspect_ratio_window * 5) + 1):
            self.aspect_ratio_memory.pop(0)

        reversed_list = list(self.aspect_ratio_memory)[::-1]

        moving_average = []
        overlap = self.aspect_ratio_window  // 2
        for x in range(6):
            if len(reversed_list) >= ((x*overlap) + self.aspect_ratio_window):
                arr = self.aspect_ratio_memory
                #print('avg', x, np.mean(arr[(x*overlap):((x*overlap) + self.aspect_ratio_window)][arr != 0]), arr[(x*overlap):((x*overlap) + self.aspect_ratio_window)])
                moving_average.append( np.mean(arr[(x*overlap):((x*overlap) + self.aspect_ratio_window)][arr != 0]) )

        cur = 0
        increase = []
        for x in moving_average:
            if cur == 0:
                cur = x
            else:
                if cur > x:
                    increase.append(True)
                else:
                    increase.append(False)
                cur = x

        threshold = 0
        if np.all(np.array(increase)):
            threshold = self.aspect_ratio_lie_threshold_long
        elif np.all(np.array(increase[:3])) or np.sum(np.array(increase) == True) > len(increase) - 2:
            threshold = self.aspect_ratio_lie_threshold_short

        #print(reversed_list)
        #print('mv', moving_average, increase, threshold)

        return threshold

    def get_bounding_box_of_detected_pose(self, frame=None, pose_landmark=None, show=True):
        frame_height, frame_width = frame.shape[:2]
        #kof = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulder to ankle only
        kof = [11, 12, 23, 24, 27, 28]  # shoulder to ankle only excluding the knees
        #print(kof, pose_landmark, 'xas \n', pose_landmark[kof])

        # Initialize bounding box
        bbox = {'xmin': 1, 'ymin': 1, 'xmax': 0, 'ymax': 0}

        # Update bounding box dimensions
        if pose_landmark is not None:
            key_landmark = pose_landmark[kof]
            bbox['xmin'] = min(bbox['xmin'], key_landmark[:,0].min())
            bbox['ymin'] = min(bbox['ymin'], key_landmark[:,1].min())
            bbox['xmax'] = max(bbox['xmax'], key_landmark[:,0].max())
            bbox['ymax'] = max(bbox['ymax'], key_landmark[:,1].max())

        # Calculate bounding box parameters
        bbox_width = bbox['xmax'] - bbox['xmin']
        bbox_height = bbox['ymax'] - bbox['ymin']
        aspect_ratio = bbox_width / bbox_height if bbox_height != 0 else 0

        if show:
            # Convert bounding box coordinates to pixel values
            bbox_pixel = {
                'xmin': int(bbox['xmin'] * frame_width),
                'ymin': int(bbox['ymin'] * frame_height),
                'xmax': int(bbox['xmax'] * frame_width),
                'ymax': int(bbox['ymax'] * frame_height)
            }
            # Pink rectangle
            cv2.rectangle(
                frame,
                (bbox_pixel['xmin'], bbox_pixel['ymin']),
                (bbox_pixel['xmax'], bbox_pixel['ymax']),
                (255, 0, 255), 2
            )

        return aspect_ratio

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

    def get_timestamp(self, frame_count=None, video_fps=None, export_fps=0, show_full_info=True):
        # Calculate timestamp in seconds (with fraction for frames_per_second)
        timestamp_sec = frame_count / video_fps

        # Convert timestamp to format hh:mm:ss:ff (including frame fraction)
        hours = int(timestamp_sec // 3600)
        minutes = int((timestamp_sec % 3600) // 60)
        seconds = int(round(timestamp_sec))
        fraction = 0
        if export_fps > 0:
            fraction = int((timestamp_sec * export_fps) % export_fps)

        if show_full_info:
            timestamp_text = f"frame: {frame_count:04d} - {hours:02}:{minutes:02}:{seconds:02}.{fraction:01} - {round(timestamp_sec):.2f} - fps: {video_fps:.2f}"
        else:
            timestamp_text = f"{hours:02}:{minutes:02}:{seconds:02}"

        return timestamp_text, timestamp_sec

    def generate_frames_for_groundtruth(self, video_file=None, frames_per_second=1, BASE_OUTPUT_DIR=None):
        self.my_frame_diff = FrameDiff()
        self.my_frame_diff.output_folder = os.path.join(BASE_OUTPUT_DIR, "output_groundtruth")
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
                cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
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

    def aspect_ratio_membership(self, aspect_ratio):
        # Use of a linear function to assign probabilistic values for aspect ratio membership
        divisor = 0.5

        # When aspect_ratio is less than or equal to 0.35, the membership value is 1 (fully low).
        threshold = 0.35
        threshold = 0.18
        low = max(0, min(1, ((threshold + divisor) - aspect_ratio) / divisor))

        # Cluster 4 + (Cluster 8 - Cluster 4) / 2
        # 0.853485 + ( 1.243108 - 0.853485) / 2 = 1.05
        # 1.05 -/+ 0.5 = 0.55, 1.55
        # low_medium = max(0, min( (aspect_ratio - 0.55) / 0.5, (1.55 - aspect_ratio) / 0.5))
        threshold = 1.05
        threshold = 0.22
        low_medium = max(0, min((aspect_ratio - (threshold - divisor)) / divisor,
                                ((threshold + divisor) - aspect_ratio) / divisor))

        threshold = 1.24
        threshold = 0.25
        medium_low = max(0, min((aspect_ratio - (threshold - divisor)) / divisor,
                                ((threshold + divisor) - aspect_ratio) / divisor))

        threshold = 2.07
        threshold = 0.35
        medium = max(0, min((aspect_ratio - (threshold - divisor)) / divisor,
                            ((threshold + divisor) - aspect_ratio) / divisor))

        threshold = 2.76
        threshold = 0.76
        medium_high = max(0, min((aspect_ratio - (threshold - divisor)) / divisor,
                                 ((threshold + divisor) - aspect_ratio) / divisor))

        # High Medium threshold is calculated as a soft margin between Clusters 3 and 5
        # Cluster 3 + ( (Cluster 5 - Cluster 3) / 4 ) = 3.33
        threshold = 3.33
        threshold = 1.33
        high_medium = max(0, min((aspect_ratio - (threshold - divisor)) / divisor,
                                 ((threshold + divisor) - aspect_ratio) / divisor))

        high_threshold = 4.99
        high = max(0, min(1, (aspect_ratio - threshold) / (high_threshold - threshold)))

        return [low, low_medium, medium_low, medium, medium_high, high_medium, high]

    def predict_posture_from_aspect_ratio(self, aspect_ratio=None):
        # Use bounding box shape to make prediction
        label = None
        posture = ['stand', 'uncertain_stand', 'uncertain_sit', 'sit', 'uncertain_sit', 'lie', 'lie']
        if aspect_ratio is not None:
            aspect_ratio = self.aspect_ratio_membership(aspect_ratio)
            max_value = max(aspect_ratio)
            max_index = aspect_ratio.index(max_value)
            # print(max_index, max_value, posture[max_index])
            label = posture[max_index]

        return label


