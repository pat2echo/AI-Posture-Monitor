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
        df['label'] = df.apply(lambda row: get_groundtruth_from_image_name(image_name=row["image_name"]), axis=1)
        df['predicted_label'] = df.apply(lambda row: predict_features(features_df=row), axis=1)
        save_path = features_file.split('.')[0] + '_predicted.csv'
        df.to_csv(save_path, index=False)
        print(df.dropna(axis=0, how='any'))
    else:
        print("Specify input features file in csv format")
else:
    print("Specify input features file in csv")