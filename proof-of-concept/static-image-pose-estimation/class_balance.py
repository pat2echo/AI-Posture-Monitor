import sys
import os
import pandas as pd
from pose_estimation_dependencies import get_groundtruth_from_image_name

if len(sys.argv) > 1:
    print(sys.argv[1])
    if os.path.isfile(sys.argv[1]):
        df = pd.read_csv(sys.argv[1])
        df['label'] = df.apply(lambda row: get_groundtruth_from_image_name(image_name=row["image_name"]), axis=1)
        #print(df)
        print('class distribution', df['label'].value_counts(normalize=True) * 100 )

    else:
        print(f'File does not exist: {sys.argv[1]}')
else:
    print("Expecting csv file name of manual labels")
    exit()
