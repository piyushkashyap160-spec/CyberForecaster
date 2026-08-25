"""
NodeFeatureScaler: Normalizes GNN node features.

Node feature schema (10 dims per node):
  0: inbound_packets    -- skewed count  -> log1p then StandardScaler
  1: outbound_packets   -- skewed count  -> log1p then StandardScaler
  2: inbound_bytes      -- skewed count  -> log1p then StandardScaler
  3: outbound_bytes     -- skewed count  -> log1p then StandardScaler
  4: unique_peers       -- skewed count  -> log1p then StandardScaler
  5: unique_dst_ports   -- skewed count  -> log1p then StandardScaler
  6: connection_rate    -- skewed count  -> log1p then StandardScaler
  7: failed_connection_rate -- ratio 0-1 -> StandardScaler only
  8: syn_ratio          -- ratio 0-1     -> StandardScaler only
  9: ack_ratio          -- ratio 0-1     -> StandardScaler only

log1p is applied to the 7 positively-skewed count features (indices 0-6).
It is NOT applied to the ratio features (indices 7-9) which are already bounded [0,1].

The scaler must be fitted ONLY on training data.
Validation and test data use .transform() only.
"""

import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Feature indices where log1p pre-processing is applied (skewed counts)
LOG1P_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6]
# Ratio features (already bounded [0,1]): indices 7, 8, 9
RATIO_FEATURE_INDICES = [7, 8, 9]

NODE_FEATURE_DIM = 10


class NodeFeatureScaler:
    """
    Fits a StandardScaler on (optionally log1p-transformed) node feature matrices.
    Designed to be fitted on training node features only.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        """
        Apply log1p to skewed count features (indices 0-6).
        X shape: (N_total_nodes, 10)
        """
        X_out = X.copy().astype(np.float32)
        X_out[:, LOG1P_FEATURE_INDICES] = np.log1p(X_out[:, LOG1P_FEATURE_INDICES])
        return X_out

    def fit(self, X_list: list) -> "NodeFeatureScaler":
        """
        Fit on a list of node feature matrices from the training split.
        X_list: list of np.ndarray with shape (num_nodes, 10) per window.
        All node rows are concatenated across windows before fitting.
        """
        all_rows = np.concatenate(X_list, axis=0)  # (total_nodes_across_training, 10)
        all_rows_log = self._apply_log1p(all_rows)
        self.scaler.fit(all_rows_log)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform a single node feature matrix (num_nodes, 10).
        Applies log1p then StandardScaler.
        """
        assert self.fitted, "NodeFeatureScaler must be fitted before calling transform()."
        X_log = self._apply_log1p(X)
        return self.scaler.transform(X_log).astype(np.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"NodeFeatureScaler saved to {path}")

    def load(self, path: str) -> "NodeFeatureScaler":
        self.scaler = joblib.load(path)
        self.fitted = True
        print(f"NodeFeatureScaler loaded from {path}")
        return self
