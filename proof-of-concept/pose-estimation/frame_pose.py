import cv2
import sys, os
import mediapipe as mp
import numpy as np

from frame_pose_dependencies import get_frame, frame_diff, merge_rectangles, get_area_of_interest, save_image, sort_rectangles, empty_folder

BASE_OUTPUT_DIR = "output"


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

# Get the FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# Initialize processing frame interval; set to 0 use recording speed
processing_interval = 0
if fps > 59:
    # limit to 30fps in video is > 30fps
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
output_folder = os.path.join(BASE_OUTPUT_DIR,"output_pose")
empty_folder(output_folder)

output_folder_aoi = os.path.join(BASE_OUTPUT_DIR,"output_aoi")
empty_folder(output_folder_aoi)

output_folder_aoi_pose = os.path.join(BASE_OUTPUT_DIR,"output_aoi_pose")
empty_folder(output_folder_aoi_pose)

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
                    save_image(aoi, os.path.join(output_folder_aoi, f"object_{frame_count:04d}_{x}_{y}" ))

    if process_image and len(rectangles) > 0:
        # Get area of interest from the biggest frame
        aoi_for_pose = None
        for rect in rectangles:
            (x, y, w, h) = rect
            aoi_for_pose = frame_output[y:y + h, x:x + w]
            break

        if aoi_for_pose is not None and aoi_for_pose.size > 0:
            #aoi_for_pose = cv2.cvtColor(aoi_for_pose, cv2.COLOR_GRAY2RGB)
            results = pose.process(aoi_for_pose)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    aoi_for_pose,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 117, 66), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 0), thickness=1, circle_radius=2)
                )

            # Save aoi for pose
            if frame_count % save_interval == 0:
                save_image(aoi_for_pose, os.path.join(output_folder_aoi_pose, f"pose_{frame_count:04d}"))
    else:
        #frame_output = frame_rgb
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
np.savetxt(os.path.join(output_folder, 'frame_max_values.csv'), np.array(output_data), fmt='%s', delimiter=',', header='file_name,max_value,process_image', comments='')

# Release resources
cap.release()
cv2.destroyAllWindows()