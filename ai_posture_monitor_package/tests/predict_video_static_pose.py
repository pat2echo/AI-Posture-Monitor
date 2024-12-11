import ai_posture_monitor as pm
import os, sys

BASE_OUTPUT_DIR = "output"

video_file = None
label_file = None

scaling_factor = 0.8
predict = True

if len(sys.argv) > 1:
    video_file = sys.argv[1]

    if len(sys.argv) > 3:
        scaling_factor = float(sys.argv[3])

    if len(sys.argv) > 4:
        label_file = sys.argv[4]

else:
    print("Expecting 2 arguments: video file and scaling factor")
    exit()

print('video_file', video_file, 'label_file', label_file, 'is_predict_pose', predict, 'scaling_factor', scaling_factor)

pe = pm.PoseEstimation()
pe.process_video(video_file=video_file, label_file=label_file, is_predict_pose=predict, model_number=2, use_frame_diff=True, use_bounding_box=True, scaling_factor=scaling_factor, BASE_OUTPUT_DIR=BASE_OUTPUT_DIR, plot_results=True, predict_fall=False)

#print(pe.get_fall_prediction(data_array=data))
