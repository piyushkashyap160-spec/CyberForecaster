import numpy as np
from typing import Dict, List, Tuple

def compute_forecast_lead_time(
    y_attack_seq: np.ndarray,
    prob_predictions: np.ndarray,
    window_seconds: float = 5.0,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes the Forecast Lead Time (early warning metric):
    The time delta (in seconds) between when model attack probability first crosses
    the decision threshold and the actual onset timestamp of the attack episode.

    Positive lead time -> early prediction before attack onset.
    Zero lead time -> detection at exact attack onset.
    Negative lead time -> detection delay after attack onset.
    """
    y_attack = (y_attack_seq >= 0.5).astype(int)
    probs = np.array(prob_predictions, dtype=float)

    # 1. Identify contiguous attack episodes (start_idx, end_idx)
    episodes: List[Tuple[int, int]] = []
    in_episode = False
    start_idx = 0

    for i in range(len(y_attack)):
        if y_attack[i] == 1 and not in_episode:
            in_episode = True
            start_idx = i
        elif y_attack[i] == 0 and in_episode:
            in_episode = False
            episodes.append((start_idx, i - 1))

    if in_episode:
        episodes.append((start_idx, len(y_attack) - 1))

    if not episodes:
        return {
            "mean_lead_time_seconds": 0.0,
            "median_lead_time_seconds": 0.0,
            "max_lead_time_seconds": 0.0,
            "episodes_detected": 0,
            "total_episodes": 0
        }

    lead_times_seconds: List[float] = []
    detected_count = 0

    for onset_idx, end_idx in episodes:
        # Search backwards and forwards around onset to find first threshold crossing
        search_start = max(0, onset_idx - 10)
        
        trigger_idx = None
        for idx in range(search_start, end_idx + 1):
            if probs[idx] >= threshold:
                trigger_idx = idx
                break

        if trigger_idx is not None:
            detected_count += 1
            lead_sec = float(onset_idx - trigger_idx) * window_seconds
            lead_times_seconds.append(lead_sec)

    if not lead_times_seconds:
        return {
            "mean_lead_time_seconds": 0.0,
            "median_lead_time_seconds": 0.0,
            "max_lead_time_seconds": 0.0,
            "episodes_detected": 0,
            "total_episodes": len(episodes)
        }

    return {
        "mean_lead_time_seconds": round(float(np.mean(lead_times_seconds)), 2),
        "median_lead_time_seconds": round(float(np.median(lead_times_seconds)), 2),
        "max_lead_time_seconds": round(float(np.max(lead_times_seconds)), 2),
        "episodes_detected": detected_count,
        "total_episodes": len(episodes)
    }
