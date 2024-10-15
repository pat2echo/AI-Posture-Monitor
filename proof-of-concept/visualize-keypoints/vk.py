# Visualize Keypoints, sources could be openposes.com or others
# this is the first step in mapping relationships that would be used to validate activity recognition algorithm
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import json

def get_open_pose_connections():
    # openpose connectors map
    # Cao, Z., Hidalgo, G., Simon, T., Wei, S. E., & Sheikh, Y. (2021). OpenPose: Realtime multi-person 2D pose estimation using Part Affinity Fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(1), 172-186.
    # https://ieeexplore.ieee.org/document/8765346
    # (2) COCO keypoint challenge dataset [67],
    # which requires simultaneously detecting people and localizing 17 keypoints (body parts)
    # in each person (including 12 human body parts and 5 facial keypoints)

    # Andriluka, M., Pishchulin, L., Gehler, P., & Schiele, B. (2014). 2D Human Pose Estimation: New Benchmark and State of the Art Analysis. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    # https://ieeexplore.ieee.org/document/6909866

    connections = [
        (0, 1),  # Nose to Neck
        (1, 2),  # Neck to left shoulder
        (1, 5),  # Neck to right shoulder
        (2, 3), (3, 4),  # Left arm
        (5, 6), (6, 7),  # Right arm
        (1, 8), (8, 9), (9, 10),  # Neck to Hip to Left leg
        (1, 11), (11, 12), (12, 13),  # Neck to Hip to Right leg
        (0, 14), (0, 15),  # Nose to eyes
        (14, 16), (15, 17)  # Eyes to ears
    ]

    return connections

def read_json_file(filename):
    # Open and read the JSON file
    json_data = None
    with open(filename, 'r') as file:
        json_data = json.load(file)

    return json_data

def json_parse(json_data=None, keypoints_count=18):
    # Retrieve canvas dimensions and pose landmarks from JSON String

    # Load your JSON data
    if json_data is None:
        json_data = read_json_file( os.path.join( os.path.dirname(os.path.realpath(__file__)), 'test.json') )

    # Extract keypoints and canvas dimensions
    keypoints = json_data[0]['people'][0]['pose_keypoints_2d']
    canvas_height = json_data[0]['canvas_height']
    canvas_width = json_data[0]['canvas_width']

    return keypoints, canvas_height, canvas_width

def plot_keypoints(keypoints, connections):
    # Prepare points for plotting
    points = []
    confidence_list = []
    for i in range(0, len(keypoints), 3):
        x, y, confidence = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        confidence_list.append(confidence)
        points.append((x, y))

    points = np.array(points)

    # Create a figure and axis
    plt.figure(figsize=(8, 8))
    plt.xlim(0, canvas_width)
    plt.ylim(canvas_height, 0)  # Invert y-axis for image coordinates

    # Plot keypoints
    plt.scatter(points[:, 0], points[:, 1], c='green', s=5, label='Keypoints')

    for index, (start, end) in enumerate(connections):
        # Optional: filter based on confidence
        if confidence_list[index] > 0.1:
            plt.plot(points[[start, end], 0], points[[start, end], 1], color='blue')

            # Label the start and end points exclude keypoints in the head
            if start < 14 and end < 14:
                plt.text(points[start, 0], points[start, 1], f'{start}', fontsize=12, ha='right', color='green')
                plt.text(points[end, 0], points[end, 1], f'{end}', fontsize=12, ha='left', color='red')

    # Set labels and title
    plt.title('Visualization of body keypoints downloaded from openposes.com')
    plt.suptitle('Pose Landmarks')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.legend()
    plt.grid()
    plt.show()

json_data = None
if len (sys.argv) > 1:
    print(sys.argv[1])
    json_data = read_json_file( sys.argv[1] )

keypoints, canvas_height, canvas_width = json_parse(json_data=json_data)
connections = get_open_pose_connections()
plot_keypoints(keypoints, connections)
