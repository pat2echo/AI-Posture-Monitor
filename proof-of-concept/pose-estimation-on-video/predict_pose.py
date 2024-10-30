import sys

from pose_est_dependencies import PoseEstimation

BASE_OUTPUT_DIR = "output"

video_file = None
# video_file = "walking_to_sit.mp4"

scaling_factor = 0.5
predict = False

if len(sys.argv) > 1:
    print(sys.argv[1])
    video_file = sys.argv[1]

    if len(sys.argv) > 2 and int(sys.argv[2]) > 0:
        predict = True

    if len(sys.argv) > 3:
        scaling_factor = float(sys.argv[3])
else:
    print("Expecting 2 arguments: video file and scaling factor")
    exit()

pe = PoseEstimation()
pe.process_video(video_file=video_file, is_predict_pose=predict, model_number=2, use_frame_diff=True, use_bounding_box=False, scaling_factor=scaling_factor, BASE_OUTPUT_DIR=BASE_OUTPUT_DIR)
