import numpy as np
import pandas as pd
from typing import Dict, List
from preprocessing.state_encoder import STATE_FEATURE_KEYS

def compute_shap_kernel_attributions(model_predict_fn, sample_sequence: np.ndarray, background_samples: np.ndarray) -> List[Dict]:
    """
    Computes SHAP values using KernelExplainer if available.
    Fallback to perturbation sensitivity analysis if SHAP is not present or slow.
    """
    try:
        import shap
        # Flatten background samples to 2D for KernelExplainer
        L, D = sample_sequence.shape
        bg_flat = background_samples.reshape(background_samples.shape[0], -1)
        sample_flat = sample_sequence.reshape(1, -1)

        explainer = shap.KernelExplainer(model_predict_fn, bg_flat[:20])
        shap_vals = explainer.shap_values(sample_flat)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_last_step = shap_vals[0].reshape(L, D)[-1, :]

        results = []
        for i, key in enumerate(STATE_FEATURE_KEYS):
            results.append({
                'feature': key,
                'shap_value': float(shap_last_step[i]),
                'abs_importance': float(abs(shap_last_step[i]))
            })
        return sorted(results, key=lambda x: x['abs_importance'], reverse=True)
    except Exception as e:
        print(f"SHAP KernelExplainer fallback: {e}")
        # Perturbation fallback
        base_pred = model_predict_fn(sample_sequence.reshape(1, -1))[0]
        results = []
        sample_flat = sample_sequence.reshape(-1)
        for i in range(len(sample_flat)):
            feat_idx = i % D
            key = STATE_FEATURE_KEYS[feat_idx]
            temp = sample_flat.copy()
            temp[i] += 0.1 * (abs(temp[i]) + 1e-5)
            new_pred = model_predict_fn(temp.reshape(1, -1))[0]
            diff = float(new_pred - base_pred)
            results.append({
                'feature': key,
                'shap_value': diff,
                'abs_importance': abs(diff)
            })
        return sorted(results, key=lambda x: x['abs_importance'], reverse=True)[:len(STATE_FEATURE_KEYS)]
