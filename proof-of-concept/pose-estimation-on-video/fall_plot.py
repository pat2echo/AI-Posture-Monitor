import sys
import os
from plot_dependencies import plot_fall
from evaluation_dependencies import Evaluate

if len(sys.argv) > 1:
    print(sys.argv[1])
    if os.path.isfile(sys.argv[1]):
        res, res2 = plot_fall(csv_file=sys.argv[1], plot_title='fall_'+os.path.basename(sys.argv[1]).split('.')[0])
        eval = Evaluate()
        print(eval.calculate_metrics(df=res))
        print(res2)
        print(eval.calculate_metrics(df=res2))

        print('class distribution', res2['label'].value_counts(normalize=True) * 100 )
        #print(eval.calculate_metrics2(df=res))
    else:
        print(f'File does not exist: {sys.argv[1]}')
else:
    print("Expecting csv file name of manual labels")
    exit()
