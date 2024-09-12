#############
# Resources
#############
# inspired by: https://infoaryan.com/blog/opencv-python-object-tracking-frame-differencing-algorithm-project/
# test video from: https://www.pexels.com/video/a-man-walking-in-orange-jumpsuit-9222270/

import sys, cv2

# Function to capture and resize frames
def get_frame(cap, scaling_factor=None, res=None):
    def get_frame_scale_down(cap, scaling_factor):
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Function to capture and resize frames to 480x600
    def get_frame_resized(cap, res):
        ret, frame = cap.read()
        if not ret:
            return None
        # Resize the frame to 480x600 resolution
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


# Function to get bounding boxes of moving objects
def get_bounding_boxes(thresh_frame, original_frame):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small changes/noise
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a bounding box around the detected region
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_frame


def get_bounding_boxes_pad_ratio(thresh_frame, original_frame, padding_ratio=0.1):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the dimensions of the original frame (for clamping)
    height, width = original_frame.shape[:2]

    # Loop through each contour and draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small changes/noise
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate dynamic padding based on a percentage of width and height
            padding_w = int(w * padding_ratio)
            padding_h = int(h * padding_ratio)

            # Add dynamic padding to the bounding box
            x_padded = max(0, x - padding_w)
            y_padded = max(0, y - padding_h)
            w_padded = min(width - x_padded, w + 2 * padding_w)
            h_padded = min(height - y_padded, h + 2 * padding_h)

            # Draw the padded bounding box
            cv2.rectangle(original_frame,
                          (x_padded, y_padded),
                          (x_padded + w_padded, y_padded + h_padded),
                          (0, 255, 0), 2)

    return original_frame


# Function to get bounding boxes of moving objects using connected components
def get_bounding_boxes_connected_components(thresh_frame, original_frame):
    # Find connected components and their stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_frame)

    # Loop through each component and draw bounding boxes, ignoring the background
    for i in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[i]
        if area > 500:  # Filter out small components
            # Draw a bounding box around the detected region
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_frame


def get_bounding_boxes_connected_components_padded(thresh_frame, original_frame, padding=10):
    # Find connected components and their stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_frame)

    # Get the dimensions of the original frame (for clamping)
    height, width = original_frame.shape[:2]

    # Loop through each component and draw bounding boxes, ignoring the background
    for i in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[i]
        if area > 500:  # Filter out small components

            # Add padding to the bounding box
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(width - x_padded, w + 2 * padding)
            h_padded = min(height - y_padded, h + 2 * padding)

            # Draw the padded bounding box
            cv2.rectangle(original_frame,
                          (x_padded, y_padded),
                          (x_padded + w_padded, y_padded + h_padded),
                          (0, 255, 0), 2)

    return original_frame


def get_bounding_boxes_connected_components_pad_ratio(thresh_frame, original_frame, padding_ratio=0.1):
    # Find connected components and their stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh_frame)

    # Get the dimensions of the original frame (for clamping)
    height, width = original_frame.shape[:2]

    # Loop through each component and draw bounding boxes, ignoring the background
    for i in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[i]
        if area > 500:  # Filter out small components

            # Calculate dynamic padding based on a percentage of width and height
            padding_w = int(w * padding_ratio)
            padding_h = int(h * padding_ratio)

            # Add dynamic padding to the bounding box
            x_padded = max(0, x - padding_w)
            y_padded = max(0, y - padding_h)
            w_padded = min(width - x_padded, w + 2 * padding_w)
            h_padded = min(height - y_padded, h + 2 * padding_h)

            # Draw the padded bounding box
            cv2.rectangle(original_frame,
                          (x_padded, y_padded),
                          (x_padded + w_padded, y_padded + h_padded),
                          (0, 255, 0), 2)

    return original_frame

def main(video_file=None, scaling_factor=1):
    if video_file is None:
        print("Video file was not specified")
        return
    # Capture frames from the video file
    cap = cv2.VideoCapture(video_file)
    #scaling_factor = 0.2 #1.5
    res = None
    #res = (640,480)

    # Capture the first three frames
    prev_frame = get_frame(cap, scaling_factor, res)
    cur_frame = get_frame(cap, scaling_factor, res)
    next_frame = get_frame(cap, scaling_factor, res)

    # Iterate until the video ends or the user presses the ESC key
    while True:
        if prev_frame is None or cur_frame is None or next_frame is None:
            break  # Break if frames are not read properly (end of video)

        frame_difference = frame_diff(prev_frame, cur_frame, next_frame)
        _, frame_th = cv2.threshold(frame_difference, 0, 255, cv2.THRESH_TRIANGLE)


        ret, read_frame = cap.read()
        if not ret or read_frame is None:
            print("Video has ended.")
            break  # Exit the loop or handle the error as needed

        # Get the bounding boxes of changed regions in the original frame (non-grayscale)
        if scaling_factor is not None:
            original_frame = cv2.resize(read_frame, None, fx=scaling_factor, fy=scaling_factor,
                                        interpolation=cv2.INTER_AREA)
        else:
            original_frame = cv2.resize(read_frame, res, interpolation=cv2.INTER_AREA)


        #output_frame = get_bounding_boxes(frame_th, original_frame)
        #output_frame = get_bounding_boxes_connected_components(frame_th, original_frame)
        #output_frame = get_bounding_boxes_connected_components_padded(frame_th, original_frame, padding=50)
        output_frame = get_bounding_boxes_connected_components_pad_ratio(frame_th, original_frame, padding_ratio=0.8)
        #output_frame = get_bounding_boxes_pad_ratio(frame_th, original_frame, padding_ratio=0.8)

        # Display the results
        #cv2.imshow("Object Movement", frame_difference)
        cv2.imshow("Object Movement", output_frame)
        #cv2.imshow("Thresholded Image", frame_th)


        # Update frames
        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap, scaling_factor)

        # Check for the ESC key press
        key = cv2.waitKey(30)  # Adjust the wait time for smoother video playback
        if key == 27:  # ESC key
            break

    # Release the video file and close windows
    cap.release()
    cv2.destroyAllWindows()

if len (sys.argv) > 1:
    print(sys.argv[1])
    video_file = sys.argv[1]
    scaling_factor = float(sys.argv[2])

main(video_file = video_file, scaling_factor=scaling_factor)

