from typing import List, Optional

def calculate_forecast_lead_time(
    predicted_probabilities: List[float],
    actual_attack_labels: List[int],
    warning_threshold: float = 0.70,
    window_seconds: float = 5.0
) -> Optional[float]:
    """
    Calculates forecast lead time in seconds:
    Delta between the timestamp when model forecast crosses warning threshold
    and the timestamp when actual attack progression manifests.
    """
    first_warning_idx = None
    first_attack_idx = None

    for idx, prob in enumerate(predicted_probabilities):
        if first_warning_idx is None and prob >= warning_threshold:
            first_warning_idx = idx

    for idx, label in enumerate(actual_attack_labels):
        if first_attack_idx is None and label == 1:
            first_attack_idx = idx

    if first_warning_idx is not None and first_attack_idx is not None:
        lead_steps = first_attack_idx - first_warning_idx
        return max(0.0, float(lead_steps * window_seconds))

    return None
