import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error

def compute_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes Precision, Recall, F1, and False Positive Rate (FPR).
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, (fp + tn)))

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'fpr': round(fpr, 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes MSE and MAE for state vector prediction S(t+1).
    """
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        'mse': round(mse, 6),
        'mae': round(mae, 6)
    }
