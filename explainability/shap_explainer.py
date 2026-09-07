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
    Computes d(Target) / d(S(t, f)) across all sequence windows for fast SOC dashboard explanations.

    Explicitly targets:
      - 'attack_probability' (default): d(P_attack) / d(S)
      - 'stage_logit': d(Logit_stage) / d(S) for specific MITRE stage
    """
    def __init__(self, model: torch.nn.Module, scaler: StateScaler):
        self.model = model
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def explain_instance(
        self,
        sequence: np.ndarray,
        target: str = "attack_probability",
        target_stage_id: int = None
    ) -> Dict:
        """
        Computes gradient saliency feature attribution for input sequence (L, 23).
        Preserves temporal timestep positioning across all L windows.
        """
        if sequence.ndim == 1:
            sequence = sequence[np.newaxis, :]  # (1, 23) -> handle single vector
        if sequence.shape[0] < 10:
            # Repeat or pad to sequence length if necessary
            sequence = np.pad(sequence, ((10 - sequence.shape[0], 0), (0, 0)), mode="edge")

        L = sequence.shape[0]
        seq_scaled = self.scaler.transform(sequence[np.newaxis, :, :])  # (1, L, 23)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, requires_grad=True).to(self.device)

        # Forward pass
        self.model.eval()
        if hasattr(self.model, 'forward_graph_sequence'):
            dummy_g = torch.zeros((1, L, 64), device=self.device)
            pred_state, attack_prob, stage_logits = self.model.forward_graph_sequence(seq_tensor, dummy_g)
        else:
            pred_state, attack_prob, stage_logits = self.model(seq_tensor)

        # Target selection for attribution
        self.model.zero_grad()
        if target == "stage_logit" and target_stage_id is not None:
            loss_target = stage_logits[0, target_stage_id]
            target_name = f"stage_logit_{target_stage_id}"
        else:
            loss_target = attack_prob[0, 0]
            target_name = "attack_probability"

        # Backward pass for spatio-temporal gradients
        loss_target.backward()

        grads = seq_tensor.grad.detach().cpu().numpy()[0]  # (L, 23)
        raw_attributions = grads * seq_scaled[0]  # (L, 23)
        norm = np.linalg.norm(raw_attributions)
        if norm > 0:
            norm_attributions = raw_attributions / norm
        else:
            norm_attributions = raw_attributions

        # 1. Spatio-temporal attribution items with explicit relative offsets
        temporal_items = []
        for t in range(L):
            step_str = "t" if t == (L - 1) else f"t-{(L - 1 - t) * 5}s"
            for j, key in enumerate(STATE_FEATURE_KEYS):
                attr_val = float(norm_attributions[t, j])
                temporal_items.append({
                    'feature': key,
                    'timestep': step_str,
                    'window_index': t,
                    'relative_step': t - (L - 1),
                    'label': f"{key} @ {step_str}",
                    'attribution': round(attr_val, 6),
                    'abs_importance': round(abs(attr_val), 6),
                    'direction': "increases_risk" if attr_val > 1e-4 else ("decreases_risk" if attr_val < -1e-4 else "neutral"),
                    'scaled_value': float(seq_scaled[0, t, j]),
                    'original_value': float(sequence[t, j])
                })

        temporal_items_sorted = sorted(temporal_items, key=lambda x: x['abs_importance'], reverse=True)

        # 2. Feature-level aggregate (for backward-compatibility with existing top_features schema)
        feat_signed = np.sum(norm_attributions, axis=0)  # (23,)
        feat_abs = np.sum(np.abs(norm_attributions), axis=0)  # (23,)
        feature_importance = []
        for j, key in enumerate(STATE_FEATURE_KEYS):
            attr_j = float(feat_signed[j])
            feature_importance.append({
                'feature': key,
                'attribution': attr_j,
                'abs_importance': float(feat_abs[j]),
                'direction': "increases_risk" if attr_j > 1e-4 else ("decreases_risk" if attr_j < -1e-4 else "neutral"),
                'scaled_value': float(seq_scaled[0, -1, j]),
                'original_value': float(sequence[-1, j])
            })

        feature_importance = sorted(feature_importance, key=lambda x: x['abs_importance'], reverse=True)

        # 3. Temporal window-level aggregate importance
        win_abs = np.sum(np.abs(norm_attributions), axis=1)  # (L,)
        window_importance = []
        for t in range(L):
            step_str = "t" if t == (L - 1) else f"t-{(L - 1 - t) * 5}s"
            window_importance.append({
                'timestep': step_str,
                'window_index': t,
                'importance': round(float(win_abs[t]), 6)
            })

        # 4. Deterministic summary sentence synthesized directly from attribution values
        top_positive = [item for item in temporal_items_sorted if item['direction'] == "increases_risk"][:3]
        if top_positive:
            drivers_desc = ", ".join([f"{item['feature']} ({item['timestep']})" for item in top_positive])
            summary_text = f"Forecast risk driven primarily by elevated {drivers_desc}."
        else:
            summary_text = "Forecast risk indicates nominal baseline features with no dominant elevation."

        return {
            'method': 'Gradient Saliency Attribution (Fast Real-Time)',
            'target': target_name,
            'attack_probability': float(attack_prob.detach().cpu().numpy()[0, 0]),
            'predicted_stage_id': int(torch.argmax(stage_logits, dim=1).detach().cpu().numpy()[0]),
            'summary': summary_text,
            'top_temporal_features': temporal_items_sorted[:10],
            'top_features': feature_importance[:10],  # Backwards compatibility
            'all_features': feature_importance,
            'window_importance': window_importance
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
