import ai_posture_monitor as pm
import os, sys

BASE_OUTPUT_DIR = None

video_file = None
label_file = None

scaling_factor = 0.8
predict = True
debug_mode = True

if len(sys.argv) > 1:
    video_file = sys.argv[1]

    if video_file == '0' or video_file == '' or video_file == '-':
        video_file = None
    elif not os.path.isfile(video_file):
        print("Video file not found! Switching to webcam")
        video_file = None

    if len(sys.argv) > 2 and int(sys.argv[2]) > 0:
        predict = True

    if len(sys.argv) > 3:
        scaling_factor = float(sys.argv[3])

    if len(sys.argv) > 4:
        label_file = sys.argv[4]
        if video_file == '0' or video_file == '' or video_file == '-' or not os.path.exists(label_file):
            label_file = None

    if len(sys.argv) > 5:
        demo = sys.argv[5]
        if not (demo == '0' or demo == '' or demo == '-'):
            debug_mode = False

else:
    print("Expecting 2 arguments: video file and scaling factor")
    exit()

print('video_file', video_file, 'label_file', label_file, 'is_predict_pose', predict, 'scaling_factor', scaling_factor)

pe = pm.PoseEstimation()
pe.process_video(debug_mode=debug_mode, video_file=video_file, label_file=label_file, is_predict_pose=predict, model_number=2, use_frame_diff=True, use_bounding_box=True, scaling_factor=scaling_factor, BASE_OUTPUT_DIR=BASE_OUTPUT_DIR, plot_results=True, predict_fall=False)