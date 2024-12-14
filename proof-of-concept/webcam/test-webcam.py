import sys, cv2
import os
import numpy as np

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


def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)


def main(video_file=None, scaling_factor=1):
    if video_file is None:
        print("Video file was not specified, using webcam")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file)

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    res = None

    # Create a folder to save images
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)

    prev_frame = get_frame(cap, scaling_factor, res)
    cur_frame = get_frame(cap, scaling_factor, res)
    next_frame = get_frame(cap, scaling_factor, res)

    frame_count = -1
    save_interval = 30  # Save every 30th frame
    processing_interval = 0 # process video at 30fps, set to 0 use recording speed
    max_abs_threshold = 56    # absolute values above this are processed set to 0 to ignore
    data = []

    while True:
        if prev_frame is None or cur_frame is None or next_frame is None:
            break

        frame_count += 1
        process_image = True

        # Process frames at specific intervals
        if processing_interval and frame_count % processing_interval != 0:
            process_image = False

        # Get frame difference
        frame_difference = frame_diff(prev_frame, cur_frame, next_frame)

        # Get max value of absolute difference
        max_value = np.max(frame_difference)
        if max_abs_threshold and max_value < max_abs_threshold:
            process_image = False

        if process_image:
            _, frame_th = cv2.threshold(frame_difference, 0, 255, cv2.THRESH_TRIANGLE)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(frame_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a color version of the current frame for drawing
        cur_frame_color = cv2.cvtColor(cur_frame, cv2.COLOR_GRAY2BGR)
        #cur_frame_color = cur_frame
        #cur_frame_color = frame_difference

        if process_image:
            # Draw rectangles around moving areas
            for contour in contours:
                if cv2.contourArea(contour) > 800:  # Adjust this threshold as needed
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(cur_frame_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the results
        cv2.imshow("Object Movement", cur_frame_color)

        # Save frame at regular intervals
        if frame_count % save_interval == 0:
            frame_title = f"frame_{frame_count:04d}.jpg"
            output_path = os.path.join(output_folder, frame_title)
            cv2.imwrite(output_path, cur_frame_color)

            # Save max value of absolute difference to csv
            print(frame_title, max_value, process_image)
            data.append([frame_title, max_value, process_image])

        # Update frames
        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap, scaling_factor)

        # Check for the ESC key press
        key = cv2.waitKey(30)  # Adjust the wait time for smoother video playback
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

scaling_factor = 1
main(scaling_factor=scaling_factor)