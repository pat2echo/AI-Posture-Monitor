import sys
import os
from plot_dependencies import plot_transitions
from evaluation_dependencies import Evaluate

if len(sys.argv) > 1:
    print(sys.argv[1])
    res = plot_transitions(csv_file=sys.argv[1], plot_title='trans_'+os.path.basename(sys.argv[1]).split('.')[0])
    eval = Evaluate()
    print(eval.calculate_metrics(df=res))
else:
    print("Expecting csv file name of manual labels")
    exit()
