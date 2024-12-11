import ai_posture_monitor as pm
import os

plot_file = os.path.join( os.path.dirname(os.path.realpath(__file__)), 'fall_results.csv' )
print('plot_file: ', plot_file)
if os.path.isfile(plot_file) and plot_file.endswith('.csv'):
    res, res2 = pm.plot_fall(csv_file=plot_file, plot_title='fall_' + os.path.basename(plot_file).split('.')[0])
    # eval = Evaluate()
    # print(eval.calculate_metrics(df=res))
    # print(res2)
    # print(eval.calculate_metrics(df=res2))
    #
    # print('class distribution', res2['label'].value_counts(normalize=True) * 100)
else:
    print("Specify input features file in csv format")