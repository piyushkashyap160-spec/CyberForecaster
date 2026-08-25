import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class LogisticRegressionBaseline:
    """
    Logistic Regression baseline model for binary attack classification.
    Used for empirical comparison against the Temporal World Model.
    """
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits baseline model on feature vectors X (N, D).
        If 3D array (N, L, D) is passed, flattens or takes last timestep S(t).
        """
        if X.ndim == 3:
            # Use last timestep S(t) in the sequence
            X = X[:, -1, :]

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Baseline model is not fitted yet!")
        if X.ndim == 3:
            X = X[:, -1, :]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
        y_pred = self.predict(X, threshold=threshold)
        y_probs = self.predict_proba(X)

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
        print(f"Baseline Logistic Regression saved to {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Baseline model file not found at {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"Baseline Logistic Regression loaded from {filepath}")
        return self
