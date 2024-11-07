import sys
from evaluation_dependencies import Evaluate

result_file = None
if len(sys.argv) > 1:
    result_file = sys.argv[1]
else:
    print("Specify result file in csv")

eval = Evaluate()
print( eval.evaluate_static_pose_classification_on_video(result_file=result_file) )