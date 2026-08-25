"""
shap_explainer.py — Feature Attribution & Explainability Module.

Provides two distinct explainability engines:
  1. GradientSaliencyExplainer (Primary / Real-time): Fast gradient-based feature saliency attribution for SOC dashboard explanations.
  2. SHAPOfflineExplainer (Optional / Offline): Model-agnostic SHAP KernelExplainer for deep offline analysis.

Important Audit Requirement:
Gradient-based saliency is explicitly labeled as "Gradient Saliency Attribution" in the UI.
It is never mislabeled as SHAP.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler
from preprocessing.state_encoder import STATE_FEATURE_KEYS

class GradientSaliencyExplainer:
    """
    Primary real-time feature attribution engine using PyTorch gradient saliency.
    Computes d(Attack_Prob) / d(S(t)) for fast dashboard rendering.
    """
    def __init__(self, model: torch.nn.Module, scaler: StateScaler):
        self.model = model
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def explain_instance(self, sequence: np.ndarray) -> Dict:
        """
        Computes gradient saliency feature attribution for input sequence (L, 23).
        """
        seq_scaled = self.scaler.transform(sequence[np.newaxis, :, :])  # (1, L, 23)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, requires_grad=True).to(self.device)

        # Forward pass
        if hasattr(self.model, 'forward_graph_sequence'):
            # GNN fallback dummy
            dummy_g = torch.zeros((1, sequence.shape[0], 64), device=self.device)
            pred_state, attack_prob, stage_logits = self.model.forward_graph_sequence(seq_tensor, dummy_g)
        else:
            pred_state, attack_prob, stage_logits = self.model(seq_tensor)

        # Backward pass for gradients
        attack_prob.backward()

        grads = seq_tensor.grad.detach().cpu().numpy()[0]  # (L, 23)
        last_step_grads = grads[-1, :]  # (23,)
        last_step_vals = seq_scaled[0, -1, :]  # (23,)

        # Saliency attribution weight
        attributions = last_step_grads * last_step_vals
        norm = np.linalg.norm(attributions)
        if norm > 0:
            attributions = attributions / norm

        feature_importance = []
        for i, key in enumerate(STATE_FEATURE_KEYS):
            feature_importance.append({
                'feature': key,
                'attribution': float(attributions[i]),
                'abs_importance': float(abs(attributions[i])),
                'scaled_value': float(last_step_vals[i]),
                'original_value': float(sequence[-1, i])
            })

        feature_importance = sorted(feature_importance, key=lambda x: x['abs_importance'], reverse=True)

        return {
            'method': 'Gradient Saliency Attribution (Fast Real-Time)',
            'attack_probability': float(attack_prob.detach().cpu().numpy()[0, 0]),
            'top_features': feature_importance[:10],
            'all_features': feature_importance
        }


class SHAPOfflineExplainer:
    """
    Optional offline SHAP KernelExplainer for deep offline analysis.
    """
    def __init__(self, model: torch.nn.Module, scaler: StateScaler, background_data: np.ndarray):
        self.model = model
        self.scaler = scaler
        self.background_data = background_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def explain_instance(self, sequence: np.ndarray) -> Dict:
        """
        Computes offline SHAP feature values using KernelExplainer approximation.
        """
        try:
            import shap
            # Define wrapper predict function
            def predict_func(x_flat):
                # Reshape x_flat (N, 230) -> (N, 10, 23)
                N = len(x_flat)
                x_3d = x_flat.reshape(N, 10, 23)
                with torch.no_grad():
                    t = torch.tensor(x_3d, dtype=torch.float32).to(self.device)
                    if hasattr(self.model, 'forward_graph_sequence'):
                        g = torch.zeros((N, 10, 64), device=self.device)
                        _, p_att, _ = self.model.forward_graph_sequence(t, g)
                    else:
                        _, p_att, _ = self.model(t)
                    return p_att.cpu().numpy().flatten()

            bg_flat = self.scaler.transform(self.background_data).reshape(len(self.background_data), -1)[:20]
            explainer = shap.KernelExplainer(predict_func, bg_flat)
            
            target_flat = self.scaler.transform(sequence[np.newaxis, :, :]).reshape(1, -1)
            shap_values = explainer.shap_values(target_flat)[0]  # (230,)
            
            # Aggregate over timesteps per feature
            shap_2d = shap_values.reshape(10, 23)
            feat_shap = np.mean(shap_2d, axis=0)  # (23,)

            feature_importance = []
            for i, key in enumerate(STATE_FEATURE_KEYS):
                feature_importance.append({
                    'feature': key,
                    'attribution': float(feat_shap[i]),
                    'abs_importance': float(abs(feat_shap[i])),
                    'scaled_value': float(target_flat[0, -23 + i]),
                    'original_value': float(sequence[-1, i])
                })
            feature_importance = sorted(feature_importance, key=lambda x: x['abs_importance'], reverse=True)

            return {
                'method': 'SHAP KernelExplainer (Deep Offline Analysis)',
                'attack_probability': float(predict_func(target_flat)[0]),
                'top_features': feature_importance[:10],
                'all_features': feature_importance
            }
        except Exception as e:
            # Fallback to Gradient Saliency if SHAP fails or is uninstalled
            saliency = GradientSaliencyExplainer(self.model, self.scaler)
            res = saliency.explain_instance(sequence)
            res['method'] = f"Gradient Saliency Attribution (SHAP Fallback: {e})"
            return res


# Backwards compatibility alias
ModelExplainer = GradientSaliencyExplainer
