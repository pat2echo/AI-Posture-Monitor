import sys
import os
from plot_dependencies import plot_activity_fall

if len(sys.argv) > 1:
    print(sys.argv[1])
    plot_activity_fall(csv_file=sys.argv[1], plot_title=os.path.basename(sys.argv[1]))
else:
    print("Expecting csv file name of manual labels")
    exit()
