import torch
import numpy as np
from typing import List, Dict, Optional

from models.lstm_world_model import TemporalLSTMWorldModel
from preprocessing.scaler import StateScaler
from preprocessing.state_encoder import STATE_FEATURE_KEYS

def perform_k_step_rollout(
    model: TemporalLSTMWorldModel,
    scaler: StateScaler,
    historical_sequence: np.ndarray,
    k_steps: int = 5,
    device: torch.device = None,
    action: Optional[str] = "do_nothing"
) -> List[Dict]:
    """
    Performs recursive K-step forward simulation with optional counterfactual intervention action:
    [S(t-9)...S(t)] -> S(t+1) -> S(t+2) -> ... -> S(t+K)

    Input historical_sequence: numpy array of shape (L, 23) in original feature scale.
    Actions supported: 'do_nothing', 'rate_limit', 'block_port', 'isolate_host'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    rollout_results = []
    # Copy sequence in scaled domain
    current_seq_scaled = scaler.transform(historical_sequence[np.newaxis, :, :]) # (1, L, 23)

    with torch.no_grad():
        for k in range(1, k_steps + 1):
            seq_tensor = torch.tensor(current_seq_scaled, dtype=torch.float32).to(device)

            pred_state_scaled, attack_prob, stage_logits = model(seq_tensor)

            pred_state_np_scaled = pred_state_scaled.cpu().numpy() # (1, 23)
            attack_prob_val = float(attack_prob.cpu().numpy()[0, 0])
            stage_id = int(torch.argmax(stage_logits, dim=1).cpu().numpy()[0])

            # Apply action-conditioned counterfactual mitigation adjustments
            if action == "rate_limit":
                attack_prob_val = max(0.02, attack_prob_val * 0.5)
            elif action == "block_port":
                attack_prob_val = max(0.01, attack_prob_val * 0.25)
                stage_id = min(stage_id, 1)
            elif action == "isolate_host":
                attack_prob_val = 0.0
                stage_id = 0

            # Inverse scale predicted state vector back to physical telemetry domain
            pred_state_orig = scaler.inverse_transform(pred_state_np_scaled)[0]

            if action == "isolate_host":
                pred_state_orig = np.zeros_like(pred_state_orig)

            # Construct feature dict
            state_dict = {STATE_FEATURE_KEYS[i]: float(pred_state_orig[i]) for i in range(len(STATE_FEATURE_KEYS))}

            rollout_results.append({
                'horizon_step': f"t+{k}",
                'step_index': k,
                'attack_probability': round(attack_prob_val, 4),
                'predicted_stage_id': stage_id,
                'state_vector': pred_state_orig,
                'state_dict': state_dict
            })

            # Roll sliding window forward: append predicted scaled state, drop oldest
            next_step_scaled = pred_state_np_scaled[:, np.newaxis, :] # (1, 1, 23)
            current_seq_scaled = np.concatenate([current_seq_scaled[:, 1:, :], next_step_scaled], axis=1)

    return rollout_results

