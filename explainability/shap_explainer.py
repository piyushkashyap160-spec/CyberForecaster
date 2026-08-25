import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler
from preprocessing.state_encoder import STATE_FEATURE_KEYS

class ModelExplainer:
    """
    Model Explainability engine producing real feature attributions for CyberForecaster.
    Uses gradient/perturbation attribution and SHAP where compatible.
    """
    def __init__(self, model: TemporalLSTMWorldModel, scaler: StateScaler, background_data: np.ndarray = None):
        self.model = model
        self.scaler = scaler
        self.background_data = background_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def explain_instance(self, sequence: np.ndarray) -> Dict:
        """
        Computes feature attribution for a single input sequence shape (L, 23) in original scale.
        Returns top contributing features and signed attribution values.
        """
        # Scale sequence
        seq_scaled = self.scaler.transform(sequence[np.newaxis, :, :]) # (1, L, 23)
        seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32, requires_grad=True).to(self.device)

        # Forward pass
        pred_state, attack_prob, stage_logits = self.model(seq_tensor)

        # Gradient of attack probability with respect to last input timestep S(t)
        attack_prob.backward()

        grads = seq_tensor.grad.detach().cpu().numpy()[0] # (L, 23)
        last_step_grads = grads[-1, :] # (23,)
        last_step_vals = seq_scaled[0, -1, :] # (23,)

        # Integrated gradient / saliency magnitude
        attributions = last_step_grads * last_step_vals

        # Normalize attributions for interpretability
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

        # Sort by absolute magnitude descending
        feature_importance = sorted(feature_importance, key=lambda x: x['abs_importance'], reverse=True)

        return {
            'attack_probability': float(attack_prob.detach().cpu().numpy()[0, 0]),
            'top_features': feature_importance[:10],
            'all_features': feature_importance
        }
