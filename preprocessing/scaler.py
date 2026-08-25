import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Union

class StateScaler:
    """
    StandardScaler wrapper designed for 2D vectors and 3D sequence tensors (N, L, 23).
    Ensures zero data leakage by fitting strictly on training split statistics.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Fits scaler on training data X (can be shape (N, D) or (N, L, D)).
        """
        if X.ndim == 3:
            N, L, D = X.shape
            X_flat = X.reshape(-1, D)
            self.scaler.fit(X_flat)
        else:
            self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet!")
        if X.ndim == 3:
            N, L, D = X.shape
            X_flat = X.reshape(-1, D)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.reshape(N, L, D)
        else:
            return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet!")
        if X.ndim == 3:
            N, L, D = X.shape
            X_flat = X.reshape(-1, D)
            X_inv = self.scaler.inverse_transform(X_flat)
            return X_inv.reshape(N, L, D)
        else:
            return self.scaler.inverse_transform(X)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        print(f"Scaler successfully saved to {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found at {filepath}")
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        print(f"Scaler successfully loaded from {filepath}")
        return self
