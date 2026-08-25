import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class StaticLogisticRegressionBaseline:
    """
    Baseline A: Static Logistic Regression model using only the current/last network state S(t) (23 dimensions).
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, 23) or (N, L, 23). Uses last timestep S(t).
        """
        if X.ndim == 3:
            X = X[:, -1, :]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Static baseline model is not fitted yet!")
        if X.ndim == 3:
            X = X[:, -1, :]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
        y_pred = self.predict(X, threshold=threshold)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / max(1, (fp + tn)))

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Static baseline model file not found at {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
        return self


class TemporalLogisticRegressionBaseline:
    """
    Baseline B: Temporal Logistic Regression model using flattened 10-state sequence [S(t-9)...S(t)] (230 dimensions).
    Distinguishes historical feature classification from learned recurrent temporal dynamics.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, L, 23) or (N, L*23). Flattens 3D sequence to 2D matrix.
        """
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Temporal baseline model is not fitted yet!")
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
        y_pred = self.predict(X, threshold=threshold)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = float(fp / max(1, (fp + tn)))

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Temporal baseline model file not found at {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
        return self


# Backwards compatibility alias
LogisticRegressionBaseline = StaticLogisticRegressionBaseline
