import os
import sys

import pandas as pd

from pose_estimation_dependencies import predict_features, get_groundtruth_from_image_name

features_file = None
save_path = None
if len(sys.argv) > 1:
    features_file = sys.argv[1]

    if os.path.isfile(features_file) and features_file.endswith('.csv'):
        df = pd.read_csv(features_file)
        print(df)
    else:
        print("Specify input features file in csv format")
else:
    print("Specify input features file in csv")