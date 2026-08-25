import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, List
from preprocessing.state_encoder import STATE_FEATURE_KEYS

class DistributionDriftDetector:
    """
    Lightweight feature distribution drift detector.
    Compares incoming live window feature distributions against training baseline distributions.
    """
    def __init__(self, baseline_features: np.ndarray, alpha: float = 0.05):
        """
        baseline_features: array of shape (N, 23) from training split.
        """
        if baseline_features.ndim == 3:
            baseline_features = baseline_features.reshape(-1, baseline_features.shape[-1])
        self.baseline = baseline_features
        self.alpha = alpha
        self.num_features = baseline_features.shape[-1]

    def detect_drift(self, current_window_features: np.ndarray) -> Dict:
        """
        Compares incoming features (M, 23) or single window (1, 23) against baseline.
        Returns drift status, p-values per feature, and overall drift warning boolean.
        """
        if current_window_features.ndim == 3:
            current_window_features = current_window_features.reshape(-1, current_window_features.shape[-1])
        elif current_window_features.ndim == 1:
            current_window_features = current_window_features.reshape(1, -1)

        drifted_features = []
        feature_p_values = {}

        for idx in range(self.num_features):
            feat_name = STATE_FEATURE_KEYS[idx]
            base_col = self.baseline[:, idx]
            curr_col = current_window_features[:, idx]

            if len(curr_col) < 5:
                # Add slight random noise if single sample to avoid degenerate KS test
                curr_col = curr_col + np.random.normal(0, 1e-4, size=(5,))

            stat, p_val = ks_2samp(base_col, curr_col)
            feature_p_values[feat_name] = round(float(p_val), 4)

            if p_val < self.alpha:
                drifted_features.append(feat_name)

        drift_ratio = len(drifted_features) / self.num_features
        drift_warning = drift_ratio > 0.25 # Warning if > 25% features drift

        warning_message = "Distribution Stable — Network behavior matches baseline."
        if drift_warning:
            warning_message = f"⚠️ DISTRIBUTION DRIFT WARNING: {len(drifted_features)} features differ from training baseline. Model confidence may be reduced."

        return {
            'drift_warning': drift_warning,
            'warning_message': warning_message,
            'drift_ratio': round(drift_ratio, 4),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'feature_p_values': feature_p_values
        }
