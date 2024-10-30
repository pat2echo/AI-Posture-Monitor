import sys

from frame_diff_dependencies import FrameDiff

BASE_OUTPUT_DIR = "output"

video_file = None
# video_file = "walking_to_sit.mp4"

scaling_factor = 0.5
predict = False

if len(sys.argv) > 1:
    print(sys.argv[1])
    video_file = sys.argv[1]

    if len(sys.argv) > 2:
        scaling_factor = float(sys.argv[2])
else:
    print("Expecting 2 arguments: video file and scaling factor")
    exit()

fd = FrameDiff()
fd.analyze_frames(video_file=video_file, scaling_factor=scaling_factor, BASE_OUTPUT_DIR=BASE_OUTPUT_DIR)
