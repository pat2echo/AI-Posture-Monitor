import sys
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from frame_diff_dependencies import FrameDiff
from plot_dependencies import plot_histogram

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
fd.plot_histogram = 'contour_area'
fd.plot_contour_range = 1
fd.plot_contour_range = 1000
fd.plot_contour_range = 4000
fd.plot_contour_range = 12000
fd.plot_contour_range = 8000
fd.plot_contour_range = 6000
xlabel = 'Size of the Contour Area after Frame Differencing'
#fd.plot_histogram = 'bounding_box'
fd.analyze_frames(video_file=video_file, scaling_factor=scaling_factor, BASE_OUTPUT_DIR=BASE_OUTPUT_DIR)

#print(fd.plot_data)
plot_histogram( x=np.array(list(fd.plot_data.keys())), y=np.array(list(fd.plot_data.values())), title=f'Histogram of Size of Contour Area and Frequency\nRange Interval {fd.plot_contour_range}', xlabel=xlabel )
