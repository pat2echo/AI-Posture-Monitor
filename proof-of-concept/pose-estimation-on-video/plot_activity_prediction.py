import sys
import os
from plot_dependencies import plot_label_vs_prediction

show_all_classes = False
if len(sys.argv) > 1:
    print(sys.argv[1])
    if len(sys.argv) > 2 and int(sys.argv[2]) > 0:
        show_all_classes = True
    plot_label_vs_prediction(csv_file=sys.argv[1], plot_title=os.path.basename(sys.argv[1]).split('.')[0], show_all_classes=show_all_classes)
else:
    print("Expecting csv file name of manual labels")
    exit()
