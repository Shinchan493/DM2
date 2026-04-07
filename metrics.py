import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    From scratch computation of strictly requested classification metrics.
    True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN),
    F1 Score, and Balanced Accuracy
    """
    TP = TN = FP = FN = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    # Safe division helpers
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Balanced Accuracy: (Sensitivity + Specificity) / 2
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    confusion_matrix = [[TN, FP], [FN, TP]]

    return {
        "TP": TP, 
        "TN": TN, 
        "FP": FP, 
        "FN": FN,
        "F1 Score": f1,
        "Balanced Accuracy": balanced_accuracy,
        "Confusion Matrix": confusion_matrix
    }
