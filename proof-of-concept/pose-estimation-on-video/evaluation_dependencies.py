import pandas as pd
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Evaluate:
    def __init__(self):
        pass

    def evaluate_static_pose_classification_on_video(self, result_file=None):
        df = pd.read_csv(result_file)
        df["label"] = df["label"].astype(str)
        df["prediction"] = df["prediction"].astype(str)
        df["predicted_label"] = df.apply(
            lambda row: row["label"] if row["prediction"] in row["label"] else row["prediction"], axis=1)
        return self.calculate_metrics(df)

    # Function to compute accuracy, sensitivity, specificity, precision, recall, and f1-score
    def calculate_metrics(self, df):
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

