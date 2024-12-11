import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .frame_diff_dependencies import pose_landmarks

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

def detect_pose_landmarks(image_path, pose=None, show=False):
    if pose is None:
        pose, mp_drawing, mp_pose = initialize_mediapipe()

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

def display_pose_landmarks(image_path):
    pose, mp_drawing, mp_pose = initialize_mediapipe()
    results, img_rgb, df = detect_pose_landmarks(image_path, pose, show=True)

    # Copy image
    img_copy = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Ensure results.pose_landmarks is a list of NormalizedLandmarkList
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Landmarks", img_copy)
        # Wait for a key press and close the window
        cv2.waitKey(0)  # Adjust the wait time for smoother video playback
        cv2.destroyAllWindows()

def initialize_mediapipe ():
    # 2. Load MediaPipe Pose landmark estimation solution
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    # 3. Load MediaPipe Drawing Utilities
    mp_drawing = mp.solutions.drawing_utils

    return pose, mp_drawing, mp_pose

def keypoints_of_focus():
    return [11, 12, 23, 24, 25, 26, 27, 28]

def plot_pose_landmarks(landmarks_3d, plot_type='3d', show_plot=True, save_path=None, has_labels=False):
    """
    Plots pose landmarks in 2D or 3D.

    Parameters:
    - landmarks_3d: numpy array of shape (33, 3) containing pose landmarks.
    - plot_type: '2d' for 2D plotting, '3d' for 3D plotting (default is '3d').
    - show_plot: If True, the plot will be shown (default is True).
    - save_path: If provided, the plot will be saved to this path (default is None).
    """

    # Define connections between landmarks
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
        (0, 9), (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
        (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
        (12, 24), (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
        (23, 24)  # Hips
    ]

    is_labelled = {}
    show_label = []
    if has_labels:
        show_label = keypoints_of_focus()

    if plot_type == '3d':
        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot landmarks
        ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])

        # Plot connections
        for connection in CONNECTIONS:
            start, end = connection
            ax.plot([landmarks_3d[start, 0], landmarks_3d[end, 0]],
                    [landmarks_3d[start, 1], landmarks_3d[end, 1]],
                    [landmarks_3d[start, 2], landmarks_3d[end, 2]])

            # Label start and end landmarks
            if start in show_label and not is_labelled.get(start, False):
                ax.text(landmarks_3d[start, 0], landmarks_3d[start, 1], landmarks_3d[start, 2], f'{start}', fontsize=16, ha='right', color='red')
                is_labelled[start] = True
            if end in show_label and not is_labelled.get(end, False):
                is_labelled[end] = True
                ax.text(landmarks_3d[end, 0], landmarks_3d[end, 1], landmarks_3d[end, 2], f'{end}', fontsize=16, ha='left', color='blue')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('MediaPipe 3D Pose Landmarks')

        # Adjust the view angle
        ax.view_init(elev=-80, azim=-90)
        #ax.invert_yaxis()

        # Equal scaling for all axis
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    elif plot_type == '2d':
        # Create 2D plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot landmarks in 2D (using X and Y coordinates)
        ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], c='green', zorder=3, s=20, label='Keypoints')

        # Plot connections
        for connection in CONNECTIONS:
            start, end = connection
            ax.plot([landmarks_3d[start, 0], landmarks_3d[end, 0]],
                    [landmarks_3d[start, 1], landmarks_3d[end, 1]], 'b-')

            if start in show_label and not is_labelled.get(start, False):
                ax.text(landmarks_3d[start, 0], landmarks_3d[start, 1], f'{start}', fontsize=16, ha='right', color='orange')
                is_labelled[start] = True
            if end in show_label and not is_labelled.get(end, False):
                is_labelled[end] = True
                ax.text(landmarks_3d[end, 0], landmarks_3d[end, 1], f'{end}', fontsize=16, ha='left', color='red')

        ax.invert_yaxis()

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('MediaPipe 2D Pose Landmarks')

    else:
        raise ValueError("plot_type must be either '2d' or '3d'.")

    # Show or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    if show_plot:
        plt.show()

    plt.close(fig)  # Close the figure to free memory

def calculate_distance(point1, point2, dim=None ):
    # Euclidean distance
    # x1, y1, z1 = 1, 2, 3
    # x2, y2, z2 = 4, 6, 8
    # distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    #print(point1[:2], point2[:2])
    if dim == '2d':
        distance = np.linalg.norm(point1[:2] - point2[:2])
    elif dim == 'y':
        distance = np.linalg.norm(point1[1] - point2[1])
    else:
        distance = np.linalg.norm(point1 - point2)

    return distance

def calculate_percentage_difference(list1, list2, abs=True):
    # Calculate the percentage differences
    # percentage_differences = [(abs(a - b) / a) * 100 for a, b in zip(list1, list2)]
    if abs:
        return (np.abs(np.array(list1) - np.array(list2)) / np.array(list1)) * 100
    else:
        return ((np.array(list1) - np.array(list2)) / np.array(list1)) * 100

def get_features(landmarks_3d, image_name=None, model=2, return_keypoints=None):
    landmark_dict = pose_landmarks()
    landmark_dict_flipped = {v: k for k, v in landmark_dict.items()}
    # print(landmark_dict_flipped)

    def is_stand_x_pos_shoulder_eq_x_pos_knee(percentage_threshold=10):
        # X position of Shoulder ~ X position of knee ~ X position of hips
        x_pos_shoulder_knee_hip = []
        percentage_differences = []
        for x in ['left', 'right']:
            x_pos_shoulder_knee_hip.append([
                landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][0],
                landmarks_3d[landmark_dict_flipped[f'{x} hip']][0],
                landmarks_3d[landmark_dict_flipped[f'{x} knee']][0]
            ])
            distances = np.diff(np.diff(x_pos_shoulder_knee_hip[-1])) * 100
            percentage_differences.append(distances)
            #print('d', distances)

        print('x pos shoulder hip knee', x_pos_shoulder_knee_hip)
        print('x percent distance shoulder hip knee', percentage_differences)
        return np.array(percentage_differences) <= percentage_threshold

    def is_lie_shoulder_eq_hip_knee_ankle():
        # Y position of shoulder ~ ( Y position of hips || Y position of knee || Y position of ankle )
        y_shoulder = []
        for x in ['left', 'right']:
            # conditional operator has be flipped due to the inverted y-axis
            y_shoulder.append(
                np.mean( calculate_percentage_difference(
                    [
                        landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][1],
                        landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][1],
                        landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][1]]
                    ,
                    [
                        landmarks_3d[landmark_dict_flipped[f'{x} hip']][1],
                        landmarks_3d[landmark_dict_flipped[f'{x} knee']][1],
                        landmarks_3d[landmark_dict_flipped[f'{x} ankle']][1]
                    ]
                ) )
            )

        # print('is Y position of shoulder ~ ( Y position of hips || Y position of knee || Y position of ankle )', y_shoulder)
        return np.array(y_shoulder)

    def is_sit_knee_gt_hip():
        # Y position of knee ≥ Y position of hip
        knee_gt_hip = []
        for x in ['left', 'right']:
            # conditional operator has be flipped due to the inverted y-axis
            is_knee_gt = 0
            knee = landmarks_3d[landmark_dict_flipped[f'{x} knee']]
            hip = landmarks_3d[landmark_dict_flipped[f'{x} hip']]

            # test Y & Z coordinates
            if knee[1] < hip[1] and knee[2] < hip[2]:
                is_knee_gt = 5
            elif knee[1] < hip[1]:
                is_knee_gt = 2
            elif knee[2] < hip[2]:
                #distances = calculate_percentage_difference( np.abs(knee[1:]), np.abs(hip[1:]) )
                #print('d', distances)
                #distances = calculate_percentage_difference( [knee[2]], [hip[2]] )
                #print('d2', distances)
                is_knee_gt = 1

            knee_gt_hip.append(
                is_knee_gt
            )

            #distances = calculate_percentage_difference(np.abs(knee[1:]), np.abs(hip[1:]), abs=False)
            #print('d', distances)

            #print('is Y of knee ≥ Y position of hip', 1-landmarks_3d[landmark_dict_flipped[f'{x} knee']][1], 1-landmarks_3d[landmark_dict_flipped[f'{x} hip']][1])
            #print('is Z of knee ≥ Z position of hip', 1-landmarks_3d[landmark_dict_flipped[f'{x} knee']][2], 1-landmarks_3d[landmark_dict_flipped[f'{x} hip']][2])
        # print('is Y of knee ≥ Y position of hip', knee_gt_hip)
        return knee_gt_hip

    def is_sit_hip_ankle_lt_shin_bone():
        # Y distance btw hip and ankle ≤ Shin bone
        hip_ankle_lt_shin_bone = []
        for x in ['left', 'right']:
            # shin bone = knee to ankle length
            shin_bone = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} knee']],
                landmarks_3d[landmark_dict_flipped[f'{x} ankle']],
                '2d'
            )
            hip_ankle_height = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} hip']][1],
                landmarks_3d[landmark_dict_flipped[f'{x} ankle']][1]
            )

            is_hip_ankle_less = 0
            if hip_ankle_height <= shin_bone:
                is_hip_ankle_less = 5
            else:
                percentage_diff = calculate_percentage_difference(hip_ankle_height, shin_bone + (shin_bone*.33))
                #print('percentage_diff 2d', percentage_diff)
                if percentage_diff < 10:
                    is_hip_ankle_less = 2
                elif percentage_diff < 20:
                    is_hip_ankle_less = 1

            hip_ankle_lt_shin_bone.append(
                is_hip_ankle_less
            )
            #print('is hip and ankle ≤ Shin bone', np.array([hip_ankle_height] + [hip_ankle_height_3d]) <= np.array([shin_bone] + [shin_bone_3d]))
            #print('is 3d distance btw hip and ankle ≤ Shin bone', hip_ankle_height_3d, shin_bone_3d)
            #print('is Y distance btw hip and ankle ≤ Shin bone', hip_ankle_height, shin_bone)

        #print('is Y distance btw hip and ankle ≤ Shin bone', hip_ankle_lt_shin_bone)
        return hip_ankle_lt_shin_bone

    def is_sit_compare_z_hip_knee_thigh_bone():
        # Z distance btw hip and ankle ≤ Thigh bone along such axis
        if model == 1:
            return [None,None]

        z_hip_knee_thigh_bone = []
        for x in ['left', 'right']:
            hip_knee = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} hip']][0::2],
                landmarks_3d[landmark_dict_flipped[f'{x} knee']][0::2]
            )
            thigh_bone = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} hip']],
                landmarks_3d[landmark_dict_flipped[f'{x} knee']]
            )

            percentage_diff = calculate_percentage_difference(thigh_bone,hip_knee)
            #print('percentage_diff 2d', percentage_diff, hip_knee, thigh_bone)
            z_hip_knee_thigh_bone.append(percentage_diff)

        # higher values supports standing, lower values support sitting
        #print('is Z distance btw hip and ankle ≤ Thigh bone along such axis', z_hip_knee_thigh_bone)
        return z_hip_knee_thigh_bone

    def is_upright_shoulder_gt_hip_plus(percentage_threshold=10):
        # Avg distance of Y pos of shoulders > avg distance of Y pos of hips
        x = 'left'
        y = 'right'
        avg_distance = []
        shoulder_to_hip = []

        for x in ['shoulder', 'hip']:
            # avg distance btw Y pos of shoulders, hips
            # minus 1 used to compensate for inverted y-axis
            avg_distance.append(
                1 - ( ( landmarks_3d[landmark_dict_flipped[f'left {x}']][1] + landmarks_3d[landmark_dict_flipped[f'right {x}']][1] ) / 2 )
            )

        for x in ['left', 'right']:
            shoulder_to_hip.append(
                calculate_distance(
                    landmarks_3d[landmark_dict_flipped[f'{x} shoulder']],
                    landmarks_3d[landmark_dict_flipped[f'{x} hip']],
                    '3d'
                )
            )

        avg_length_should_to_hip = (shoulder_to_hip[0] + shoulder_to_hip[1]) / 2
        percentage_differences = calculate_percentage_difference(avg_distance[0], avg_distance[1], abs=False)

        is_upright = False
        percentage_upright = 100 - int((avg_length_should_to_hip - np.abs(avg_distance[0] - avg_distance[1])) / avg_length_should_to_hip * 100)
        if percentage_differences >= percentage_threshold:
            is_upright = True

        #print('percentage non upright',  percentage_upright)
        #print('is Avg distance of Y pos of shoulders > avg distance of Y pos of hips', percentage_differences, percentage_upright)
        return is_upright, percentage_upright

    def is_sit_shoulder_gt_hip_plus_angle(sine_of_angle_btw_back_and_chair=0.6428):
        # Too complicated: not being used
        # Y of shoulder > Y of hip + (Length of Shoulder to Hip * sin(40))
        # sin(40) = 0.6428
        # sin(60) = 0.866
        shoulder_gt_hip = []
        x = 'left'
        y = 'right'
        mhip = 1 - max(landmarks_3d[landmark_dict_flipped[f'{x} hip']][1], landmarks_3d[landmark_dict_flipped[f'{y} hip']][1] )
        mshl = 1 - max(landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][1], landmarks_3d[landmark_dict_flipped[f'{y} shoulder']][1] )

        # avg. distance btw both shoulders and hips
        shl_hip = ( calculate_distance(
            landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][0],
            landmarks_3d[landmark_dict_flipped[f'{x} hip']][0]
        ) + calculate_distance(
            landmarks_3d[landmark_dict_flipped[f'{y} shoulder']][0],
            landmarks_3d[landmark_dict_flipped[f'{y} hip']][0]
        ) ) / 2

        shoulder_gt_hip.append((
            mshl + calculate_distance( landmarks_3d[landmark_dict_flipped[f'{x} shoulder']][:2] , landmarks_3d[landmark_dict_flipped[f'{y} shoulder']][:2] ) / 2
            >
            ( sine_of_angle_btw_back_and_chair * shl_hip ) + mhip + calculate_distance( landmarks_3d[landmark_dict_flipped[f'{x} hip']][:2] , landmarks_3d[landmark_dict_flipped[f'{y} hip']][:2] ) / 2
        ))

        #print('is Y of shoulder > Y of hip + (Length of Shoulder to Hip * sin(40))', shoulder_gt_hip)
        return shoulder_gt_hip

    def is_lie(percentage_threshold=20):
        # Determine if position is lying
        shoulder_lying = is_lie_shoulder_eq_hip_knee_ankle()

        # Initialize the result array with 'non_lying'
        result = np.full(shoulder_lying.shape, 'non_lying', dtype=object)

        # Test for standing
        result[shoulder_lying < percentage_threshold] = 'lying'

        return result

    def is_sit():
        # Determine if position is sitting
        sit_left_right_leg = ['non_sitting', 'non_sitting']

        hip_ankle_lt_shin_bone = is_sit_hip_ankle_lt_shin_bone()
        knee_gt_hip = is_sit_knee_gt_hip()
        zx_hip_knee = is_sit_compare_z_hip_knee_thigh_bone()
        #print('zx_hip_knee', zx_hip_knee)
        #print('hip_ankle_lt_shin_bone', hip_ankle_lt_shin_bone)
        #print('knee_gt_hip', knee_gt_hip)

        # max value obtainable is 5 for each function, hence scale to 100%
        max = (100/10)

        for x, v in enumerate(knee_gt_hip):
            if v >= 2:
                sit_left_right_leg[x] = 'sitting'
            elif hip_ankle_lt_shin_bone[x] >= 2:
                sit_left_right_leg[x] = 'sitting'
            elif zx_hip_knee[x] is not None and zx_hip_knee[x] <= 10:
                sit_left_right_leg[x] = 'sitting'
            elif v == 1 and hip_ankle_lt_shin_bone[x] >= 2:
                sit_left_right_leg[x] = 'sitting'
            elif v == 1 or hip_ankle_lt_shin_bone[x] > 0:
                sit_left_right_leg[x] = 'uncertain_sitting'

            if zx_hip_knee[x] is not None:
                if zx_hip_knee[x] > 25:
                    v = v - (v*zx_hip_knee[x]/100)
                elif zx_hip_knee[x] < 10:
                    v = v * 3

            sit_left_right_leg.append( int(max * ( v + hip_ankle_lt_shin_bone[x] )) )

        return sit_left_right_leg

    def is_stand_hip_height_gt_bone_length(percentage_threshold=16, percentage_threshold_for_squat=38):
        # Y distance btw hip and ankle ≥ SUM(Thigh bone, Shin bone)
        bone_length = hip_ankle_length(dim='2d')
        bone_length_3d = hip_ankle_length(dim='3d')
        y_distance_hip_to_ankle = []

        for x in ['left', 'right']:
            y_distance_hip_to_ankle.append(
                calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} hip']][1],
                landmarks_3d[landmark_dict_flipped[f'{x} ankle']][1]
                )
            )

        # Calculate the percentage differences
        percentage_differences = calculate_percentage_difference(bone_length + bone_length_3d, y_distance_hip_to_ankle + y_distance_hip_to_ankle)
        #print('bone_length and y_distance_hip_to_ankle', bone_length, y_distance_hip_to_ankle)
        #print('percent diff hip-to-ankle', percentage_differences)
        #print('mean percent diff hip-to-ankle', np.mean(percentage_differences[::2]), np.mean(percentage_differences[1::2]) )

        #print('percent diff hip-to-ankle', percentage_differences)
        zx_hip_knee = is_sit_compare_z_hip_knee_thigh_bone()
        for x in range(2):
            if zx_hip_knee[x] is not None and zx_hip_knee[x] < 10:
                reduce = 1.2 + zx_hip_knee[x]
                percentage_differences[x] = percentage_differences[x] + (percentage_differences[x] * reduce)
                percentage_differences[x+2] = percentage_differences[x+2] + (percentage_differences[x+2] * reduce)
        #print('percent diff hip-to-ankle', percentage_differences)

        # Initialize the result array with 'non_standing'
        result = np.full(percentage_differences.shape, 'non_standing', dtype=object)

        # Test for standing
        result[percentage_differences < percentage_threshold] = 'standing'

        # Test for squatting
        result[(percentage_differences >= percentage_threshold) & (
                    percentage_differences < percentage_threshold_for_squat)] = 'squat'

        #print('initial result', result)

        final_result = []
        mean = [ 100 - int(np.mean(percentage_differences[::2])), 100 - int(np.mean(percentage_differences[1::2])) ]

        # both legs in 2d space indicates standing
        if result[0] == result[1] and mean[0] >= percentage_threshold and mean[1] >= percentage_threshold:
            final_result = list(result[:2])
        else:
            # perform comparison / voting btw 2d results and 3d results of bone length
            for x in range(2):
                if result[x] == result[x + 2]:
                    final_result.append(result[x])
                elif percentage_differences[x] <= percentage_threshold and percentage_differences[x+2] <= percentage_threshold_for_squat:
                    final_result.append(result[x])
                elif percentage_differences[x+2] <= percentage_threshold and percentage_differences[x] <= percentage_threshold_for_squat:
                    final_result.append(result[x+2])
                elif percentage_differences[x] <= ( percentage_threshold / 2 ) and mean[x] >= percentage_threshold:
                    final_result.append(result[x])
                else:
                    final_result.append('uncertain_' + result[x])

        # Return the result array
        final_result.append( mean[0] )
        final_result.append( mean[1] )

        return final_result

    def hip_ankle_length(dim=None):
        # Hip to knee length + knee to ankle length
        hip_ankle_length = []
        for x in ['left', 'right']:
            hip_knee_dist = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} hip']],
                landmarks_3d[landmark_dict_flipped[f'{x} knee']],
                dim
            )
            knee_ankle_dist = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} knee']],
                landmarks_3d[landmark_dict_flipped[f'{x} ankle']],
                dim
            )

            hip_ankle_length.append(hip_knee_dist + knee_ankle_dist)

        return hip_ankle_length

    # Calculate and visualize the lines
    def plot_lines_to_validate_distance(image_name=None, dim=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        color = {'left': 'r-', 'right': 'g-'}

        for x in ['left', 'right']:
            # Get the coordinates for the hip
            hip = landmarks_3d[landmark_dict_flipped[f'{x} hip']][:2]  # X, Y coordinates

            # Calculate hip-to-knee and knee-to-ankle distances
            hip_knee_length = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} hip']],
                landmarks_3d[landmark_dict_flipped[f'{x} knee']],
                dim
            )
            knee_ankle_length = calculate_distance(
                landmarks_3d[landmark_dict_flipped[f'{x} knee']],
                landmarks_3d[landmark_dict_flipped[f'{x} ankle']],
                dim
            )

            # Start the line at the hip
            start_x, start_y = hip

            # Calculate the knee position (drawn along the y-axis for simplicity)
            knee_x = start_x  # Keeping X constant, this can be adjusted if needed
            knee_y = start_y + hip_knee_length  # Moving downwards by the hip_knee_length

            # Plot line from hip to knee
            ax.plot([start_x, knee_x], [start_y, knee_y], color[x], label=f'{x} leg (hip to knee)')

            # Calculate the ankle position starting from knee position
            ankle_x = knee_x  # Keeping X constant
            ankle_y = knee_y + knee_ankle_length  # Moving downwards by knee_ankle_length

            # Plot line from knee to ankle
            ax.plot([knee_x, ankle_x], [knee_y, ankle_y], 'orange', label=f'{x} leg (knee to ankle)')

            # Plot the hip, knee, and ankle points for reference
            ax.scatter([start_x, knee_x, ankle_x], [start_y, knee_y, ankle_y], c='b')

        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Pose {image_name.upper()}: Line from Hip to Knee to Ankle')
        plt.legend()
        plt.show()

    # Plot the lines
    # plot_lines_to_validate_distance(image_name=image_name, dim='3d')

    # kof = keypoints_of_focus()
    # print(landmarks_3d[kof])
    # print(is_sit_knee_gt_hip())
    # print(is_sit_hip_ankle_lt_shin_bone())
    # print(is_lie_shoulder_eq_hip_knee_ankle())
    # print('is upright', is_upright_shoulder_gt_hip_plus())
    # print('stand', is_stand_hip_height_gt_bone_length())
    # print('sit', is_sit())
    # print('lie', is_lie())
    is_upright, percent_upright = is_upright_shoulder_gt_hip_plus()
    return_list = [image_name, is_upright, percent_upright] + list(is_stand_hip_height_gt_bone_length()) + list(is_sit()) + list(is_lie())

    keypoint_focus = {}
    if return_keypoints is not None:
        #kof = [11, 12, 23, 24, 27, 28]
        keypoint_focus = landmarks_3d[return_keypoints]
        # for x in ['left', 'right']:
        #     for keypoint in return_keypoints:
        #         keypoint_focus[f'{x} {keypoint}'] = landmarks_3d[landmark_dict_flipped[f'{x} {keypoint}']]
        return [return_list, keypoint_focus]
    else:
        return return_list