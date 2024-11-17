import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Function to compute accuracy, sensitivity, specificity, precision, recall, and f1-score
def calculate_metrics(df):
    # Extract the true labels and predicted labels
    y_true = df['label']
    y_pred = df['predicted_label']

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall (sensitivity), f1-score
    precision = precision_score(y_true, y_pred, average='macro')  # or 'weighted'
    recall = recall_score(y_true, y_pred, average='macro')  # Sensitivity
    f1 = f1_score(y_true, y_pred, average='macro')

    # Calculate confusion matrix to derive specificity
    # tn, fp, fn, tp = confusion_matrix(y_true == 'stand',
    #                                   y_pred == 'stand').ravel()  # Assuming 'stand' as positive class
    #
    # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

features_file = None
save_path = None
if len(sys.argv) > 1:
    features_file = sys.argv[1]

    non_match = None
    if len(sys.argv) > 2:
        non_match = sys.argv[2]

    if os.path.isfile(features_file) and features_file.endswith('.csv'):
        df = pd.read_csv(features_file)

        df.replace(-1.0, np.nan, inplace=True)

        metrics = calculate_metrics(df.dropna(axis=0, how='any'))
        print(metrics)

        if non_match is not None:
            dfa = df.dropna(axis=0, how='any')
            dfr = dfa[dfa['label'] == non_match]
            print(dfr[dfa['predicted_label'] != non_match])
    else:
        print("Specify input features file in csv format")
else:
    print("Specify input features file in csv")

