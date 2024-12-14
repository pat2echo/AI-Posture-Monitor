import cv2
import numpy as np
import os
import pandas as pd
import shutil

class FrameDiff:
    def __init__(self):
        self.output_folder = None
        self.output_folder_aoi = None
        self.output_folder_aoi_pose = None
        self.previous_frame_diff = None
        self.save_data = False

        self.quads = {}
        self.processed_frames = 0
        self.last_displayed_frame = []
        self.previous_frames = []
        self.previous_frames_limit = 3

        self.plot_data = {}
        self.plot_contour_range = 500
        self.plot_histogram = None

    # Function to capture and resize frames
    def get_frame(self, cap, scaling_factor=None, res=None):
        def get_frame_scale_down(cap, scaling_factor):
            ret, frame = cap.read()
            if not ret:
                return None, None
            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame

        def get_frame_resized(cap, res):
            ret, frame = cap.read()
            if not ret:
                return None, None
            frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame

        if scaling_factor is not None:
            return get_frame_scale_down(cap, scaling_factor)
        else:
            return get_frame_resized(cap, res)


    # Function to calculate frame difference
    def frame_diff(self, prev_frame, cur_frame, next_frame, dual_frame_difference=True):
        diff_frames1 = cv2.absdiff(next_frame, cur_frame)
        diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
        diff_frame = cv2.bitwise_and(diff_frames1, diff_frames2)

        if dual_frame_difference:
            bg_diff_frame = None
            if self.previous_frame_diff is not None:
                bg_diff_frame = cv2.absdiff(self.previous_frame_diff, diff_frame)
            self.previous_frame_diff = diff_frame
            diff_frame = bg_diff_frame

        return diff_frame

    def get_rgb_image_from_cv2(self, image_path, show=False):
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

    def pose_landmarks(self, ):
        return pose_landmarks()

    def detect_pose_landmarks(self, image_path, pose, show=False):
        img_rgb = self.get_rgb_image_from_cv2(image_path, show=False)
        results = pose.process(img_rgb)

        landmarks_data = []
        if results.pose_landmarks:
            if show:
                # Extract landmark data
                body_parts = self.pose_landmarks()
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

    def display_pose_landmarks(self, image_path, pose, mp_drawing):
        results, img_rgb, df = self.detect_pose_landmarks(image_path, pose, show=True)

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


    def merge_rectangles(self, rectangles, max_distance=50, pre_padding_percent=1, post_padding_percent=1):
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

        rectangles = [self.add_padding(rect, pre_padding_percent) for rect in rectangles]

        rectangles = self.sort_rectangles(rectangles)

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
            current = self.add_padding(current, post_padding_percent)

            merged.append(current)

        return merged

    def sort_rectangles(self, rectangles):
        # Convert rectangles to a numpy array for easier manipulation
        rectangles = np.array(rectangles)

        # Sort rectangles by area in descending order
        sorted_idx = np.argsort(-(rectangles[:, 2] * rectangles[:, 3]))
        return rectangles[sorted_idx]

    def add_padding(self, rect, padding_percent):
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

    def get_area_of_interest(self, frame, rect):
        """
        Save the portion of the frame within the given rectangle as a new image.

        :param frame: The full frame image
        :param rect: The rectangle coordinates (x, y, w, h)
        """
        x, y, w, h = rect
        return frame[y:y + h, x:x + w]

    def save_image(self, image, image_name):
        """
        Save the portion of the frame within the given rectangle as a new image.

        :param image: Image to save
        :param image_name: Name of the image
        """
        if image is not None and image.size > 0:
            cv2.imwrite(f"{image_name}.jpg", image)

    # Function to empty a folder
    def empty_folder(self, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Remove the entire folder
        os.makedirs(folder_path, exist_ok=True)  # Recreate the empty folder

    def get_bounding_box(self, diff_frame=None, frame_output=None, intersect_rectangles=True,
                         frame_count=0, save_interval=0, show_grid=False, snap_to_grid=True, show_rectangle=True):
        rectangles = []
        area_of_interest = {}
        if diff_frame is not None:
            # Threshold the difference frame
            _, thresh_frame = cv2.threshold(diff_frame, 0.5, 255, cv2.THRESH_BINARY)

            # Find contours in the threshold image
            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a black-and-white image to show detected differences
            bw_diff_image = np.zeros_like(diff_frame, dtype=np.uint8)

            # Get rectangles for all significant contours
            cas = []
            for contour in contours:
                ca = cv2.contourArea(contour)

                if ca > 0 and self.plot_histogram == 'contour_area':
                    # group contour area in range of 500
                    group_ca = ca // self.plot_contour_range
                    if group_ca > 0:
                        self.plot_data[group_ca] = self.plot_data.get(group_ca, 0) + 1

                if ca > 6000:
                    rect = cv2.boundingRect(contour)
                    rectangles.append(rect)

                    # Draw the contour onto the black-and-white image
                    #cv2.drawContours(bw_diff_image, [contour], -1, 255, -1)
                    #cas.append(ca)

            #cv2.imshow('Differences Only', bw_diff_image)
            #self.save_image(bw_diff_image, os.path.join(self.output_folder_aoi, f"bg_{frame_count:04d}"))
            #if len(cas) > 0:
            #    print(max(cas), min(cas))

            if len(rectangles) > 0:
                if intersect_rectangles:
                    # Merge close or intersecting rectangles
                    rectangles = self.merge_rectangles(rectangles, max_distance=500)
                else:
                    # Sort rectangles in desc order by area
                    rectangles = self.sort_rectangles(rectangles)

                if self.plot_histogram == 'bounding_box':
                    for rect in rectangles:
                        x, y, w, h = rect
                        box_area = w * h
                        self.plot_data[box_area] = self.plot_data.get(box_area, 0) + 1

            if snap_to_grid:
                _, __, rectangles, area_of_interest = self.draw_gridlines(frame=frame_output, num_rows=20, num_cols=20,
                                    color=(155, 155, 155), thickness=1, rect=rectangles, show_grid=show_grid)


            # Yellow: expanded area of interest taking into consideration previous area of interest and get their union if they intersect
            aoi_from_memory = False
            if 'rect' in area_of_interest and len(area_of_interest['rect']) > 0:
                x, y, w, h = area_of_interest['rect']
                aoi_color = (0, 0, 255)
                aoi_from_memory = area_of_interest['memory']
                if aoi_from_memory:
                    aoi_color = (255, 255, 0)

                if show_rectangle:
                    cv2.rectangle(frame_output, (x, y), (x + w, y + h), aoi_color, 3)

            area_of_interest['bbox_aspect_ratio'] = 0
            if aoi_from_memory == False and len(rectangles) > 0:
                for rect in rectangles:
                    x, y, w, h = rect
                    area_of_interest['bbox_aspect_ratio'] = w / h if h != 0 else 0
                    if show_rectangle:
                        # Green: bounding box around moving object
                        cv2.rectangle(frame_output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display black and white
                    #cv2.rectangle(bw_diff_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save area of interest (aoi) at regular intervals
                    if save_interval > 0 and frame_count % save_interval == 0:
                        # Get area of interest
                        #aoi = self.get_area_of_interest(frame_output, (x, y, w, h))

                        # Save aoi
                        #self.save_image(aoi, os.path.join(self.output_folder_aoi, f"object_{frame_count:04d}_{x}_{y}"))
                        pass

                    # Use the biggest rectangle
                    break

                #self.save_image(frame_output, os.path.join(self.output_folder_aoi, f"bg_{frame_count:04d}"))

        return rectangles, area_of_interest

    def draw_gridlines(self, frame, num_rows=10, num_cols=10, color=(0, 255, 0), thickness=1, rect=None, show_grid=True):
        # Get the image dimensions
        height, width = frame.shape[:2]

        # Calculate the spacing between grid lines
        row_spacing = height // num_rows
        col_spacing = width // num_cols

        if show_grid:
            # Draw horizontal grid lines
            for i in range(1, num_rows):
                y = i * row_spacing
                cv2.line(frame, (0, y), (width, y), color, thickness)

            # Draw vertical grid lines
            for j in range(1, num_cols):
                x = j * col_spacing
                cv2.line(frame, (x, 0), (x, height), color, thickness)

        quadrant = []
        area_of_interest = {}
        rectangles = []
        if rect is not None and len(rect) > 0:
            biggest = True
            for rec in rect:
                x1, y1, w, h = rec
                x2, y2 = x1 + w, y1 + h

                # Find the grid boundaries that enclose the smaller rectangle
                top_row = (y1 // row_spacing) * row_spacing
                bottom_row = ((y2 + row_spacing - 1) // row_spacing) * row_spacing
                left_col = (x1 // col_spacing) * col_spacing
                right_col = ((x2 + col_spacing - 1) // col_spacing) * col_spacing

                quadrant = [top_row, bottom_row, left_col, right_col]
                area = ( bottom_row - top_row ) * (right_col - left_col)
                # Set surrounding boundary to start from top to bottom of image
                if biggest:
                    top_row = 0
                    bottom_row = height - 1

                #quad_key = '-'.join(map(str, quadrant))
                quad_key = f'{left_col}-{right_col}'

                if len(self.previous_frames) > 0 and quad_key in self.previous_frames:
                    if self.previous_frames.count(quad_key) > 1:
                        left_col2, right_col2 = left_col, right_col
                        # Expand the area of interest to encompass the previous intersecting region
                        if len(self.last_displayed_frame) > 0 and ( ( self.last_displayed_frame[0] <= left_col and self.last_displayed_frame[2] >= right_col ) or ( self.last_displayed_frame[2] >= left_col and self.last_displayed_frame[2] <= right_col ) or ( self.last_displayed_frame[0] >= left_col and self.last_displayed_frame[0] <= right_col ) ):
                            if self.last_displayed_frame[0] < left_col:
                                left_col2 = self.last_displayed_frame[0]
                            if self.last_displayed_frame[2] > right_col:
                                right_col2 = self.last_displayed_frame[2]


                        self.last_displayed_frame = [left_col, top_row, right_col, bottom_row, left_col2, right_col2]
                        # print('prev', quad_key, self.previous_frames)
                        rectangles.append(rec)

                        # Draw the enclosing rectangle in red
                        area_of_interest = {'rect': (left_col2, 0, right_col2 - left_col2, height - 1), 'memory':False }
                    #else:
                        #print('skip1 prev', quad_key, self.previous_frames)
                #else:
                    #print('skip prev', quad_key, self.previous_frames)

                # if quad_key not in self.quads:
                #     self.quads[quad_key] = {'count': 0, 'frequency': 0, 'count_appearance': 0, 'frequency_appearance': 0, 'area':0}
                # if biggest:
                #     self.quads[quad_key]['count'] += 1
                #     self.quads[quad_key]['frequency'] = self.quads[quad_key]['count'] / self.processed_frames
                # self.quads[quad_key]['area'] = area
                # self.quads[quad_key]['count_appearance'] += 1
                # self.quads[quad_key]['frequency_appearance'] = self.quads[quad_key]['count_appearance'] / self.processed_frames


                self.previous_frames.append(quad_key)
                if len(self.previous_frames) > self.previous_frames_limit:
                    self.previous_frames.pop(0)

                biggest = False

        # Period of no movement, display rectangle from memory
        if len(rectangles) == 0 and len(self.last_displayed_frame) > 0:
            area_of_interest = {'rect':(self.last_displayed_frame[4], 0, self.last_displayed_frame[5] - self.last_displayed_frame[4], height - 1), 'memory':True}
            rectangles = [area_of_interest['rect']]

        return frame, quadrant, rectangles, area_of_interest

    def analyze_frames(self, video_file=None, scaling_factor=0.5, BASE_OUTPUT_DIR=None):
        self.output_folder = os.path.join(BASE_OUTPUT_DIR, "frame_analysis")
        self.empty_folder(self.output_folder)

        self.output_folder_aoi = os.path.join(BASE_OUTPUT_DIR, "frame_analysis_aoi")
        self.empty_folder(self.output_folder_aoi)

        cap = cv2.VideoCapture(video_file)

        # Get the FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second: {fps}")

        # Initialize frame variables
        prev_frame = None
        cur_frame = None
        next_frame = None

        diff_frame = None
        diff_frame_prev = None
        bg_diff_frame = None

        # Initialize current frame number
        frame_count = -1
        output_data = []
        max_abs_threshold = 0

        while True:
            # Get the next frame
            frame, frame_color = self.get_frame(cap, scaling_factor=scaling_factor)

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
            save_interval = 0

            # Set max absolute value to 0 in case frame was not processed
            max_value = 0
            sum_of_squares = 0

            # Initialize control variable to process image or not
            process_image = True

            # Convert frame to RGB for MediaPipe
            frame_output = cv2.cvtColor(frame_color.copy(), cv2.COLOR_BGR2RGB)

            if process_image:
                # Perform frame differencing
                diff_frame = self.frame_diff(prev_frame, cur_frame, next_frame, dual_frame_difference=True)

            if process_image:
                rectangles = None
                # Get max value of absolute difference
                max_value = np.max(diff_frame)
                sum_of_squares = np.sum(diff_frame)
                if max_abs_threshold > 0 and max_value < max_abs_threshold:
                    process_image = False

                self.processed_frames += 1
                rectangles, aoi = self.get_bounding_box(diff_frame=diff_frame, frame_output=frame_output, intersect_rectangles=True,
                                                   frame_count=frame_count, save_interval=save_interval, show_grid=True)

            # Display the result
            cv2.imshow('Motion Detection and Pose Estimation', cv2.cvtColor(frame_output, cv2.COLOR_RGB2BGR))

            # Save frame at regular intervals
            if process_image and ((save_interval == 0) or (save_interval > 0 and frame_count % save_interval == 0)):
                frame_title = f"frame_{frame_count:04d}.jpg"
                output_path = os.path.join(self.output_folder, frame_title)
                #cv2.imwrite(output_path, frame)
                #cv2.imwrite(output_path, frame_output)

                # Save max value of absolute difference to csv
                print(frame_title, max_value, sum_of_squares, process_image)
                output_data.append([frame_title, max_value, sum_of_squares, process_image])

            wait_time = int(1000 / fps)
            # fast-forward
            wait_time = 1
            key = cv2.waitKey(wait_time)

            #print('quads', quads)
            #if frame_count > 265:
            #    break

            #key = None
            if key == 27:  # ESC key
                break

        # Save the NumPy array to CSV
        if self.save_data:
            np.savetxt(os.path.join(self.output_folder, 'frame_max_values.csv'), np.array(output_data), fmt='%s', delimiter=',',
                   header='file_name,max_value,sum_of_squares,process_image', comments='')

        # Save Quadrants to CSV
        #output_data = [[key, val['count'], val['frequency'], val['count_appearance'], val['frequency_appearance'], val['area']] for key, val in self.quads.items()]
        #np.savetxt(os.path.join(self.output_folder, 'quadrants.csv'), np.array(output_data), fmt='%s',
        #           delimiter=',', header='quad,count,frequency,count_appearance,frequency_appearance,area', comments='')

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

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
