import sys

from pose_est_dependencies import PoseEstimation

BASE_OUTPUT_DIR = "output"

video_file = None
# video_file = "walking_to_sit.mp4"

scaling_factor = 0.5
predict = False
frames_per_second = 1

if len(sys.argv) > 1:
    print(sys.argv[1])
    video_file = sys.argv[1]
    if len(sys.argv) > 2:
        frames_per_second = int(sys.argv[2])
else:
    print("Expecting 1 argument: video file path")
    exit()

pe = PoseEstimation()
pe.generate_frames_for_groundtruth(video_file=video_file, BASE_OUTPUT_DIR=BASE_OUTPUT_DIR, frames_per_second=frames_per_second)
