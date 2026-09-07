import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone

def _to_timestamp_seconds(ts: Any) -> float:
    """Converts a timestamp (datetime, pd.Timestamp, str, or float) to epoch seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, pd.Timestamp):
        return ts.timestamp()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    if isinstance(ts, str):
        try:
            dt = pd.to_datetime(ts)
            return dt.timestamp()
        except Exception:
            pass
    return 0.0

def compute_forecast_lead_time(
    y_attack_seq: np.ndarray,
    prob_predictions: np.ndarray,
    window_seconds: float = 5.0,
    threshold: float = 0.5,
    timestamps: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Computes the Forecast Lead Time (early warning metric):
    The time delta (in seconds) between when model attack probability first crosses
    the decision threshold and the actual onset timestamp of the attack episode.

    - Positive lead time (> 0s) -> genuine early prediction before attack onset.
    - Zero lead time (== 0s) -> detection at exact attack onset.
    - Negative lead time (< 0s) -> detection delay after attack has already begun.

    Post-onset detections are strictly penalized with negative lead time and are
    never counted as positive pre-onset early warnings.
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
            "pre_onset_warnings": 0,
            "exact_onset_detections": 0,
            "post_onset_detections": 0,
            "total_episodes": 0
        }

    lead_times_seconds: List[float] = []
    detected_count = 0
    pre_onset_count = 0
    exact_onset_count = 0
    post_onset_count = 0

    use_real_ts = timestamps is not None and len(timestamps) >= len(y_attack)

    for ep_idx, (onset_idx, end_idx) in enumerate(episodes):
        prev_end = episodes[ep_idx - 1][1] if ep_idx > 0 else -1
        # Search backwards up to 10 windows, strictly clamped to after previous episode end
        search_start = max(0, onset_idx - 10, prev_end + 1)

        trigger_idx = None
        for idx in range(search_start, end_idx + 1):
            if probs[idx] >= threshold:
                trigger_idx = idx
                break

        if trigger_idx is not None:
            detected_count += 1
            if use_real_ts:
                t_onset = _to_timestamp_seconds(timestamps[onset_idx])
                t_trigger = _to_timestamp_seconds(timestamps[trigger_idx])
                if trigger_idx < onset_idx:
                    delta_sec = t_onset - t_trigger
                    max_allowed_delta = 10 * window_seconds + 1.0
                    if delta_sec <= max_allowed_delta:
                        lead_sec = max(0.0, float(delta_sec))
                        pre_onset_count += 1
                    else:
                        # Non-contiguous boundary jump across split blocks: not an adjacent pre-onset trigger
                        lead_sec = 0.0
                        exact_onset_count += 1
                elif trigger_idx == onset_idx:
                    lead_sec = 0.0
                    exact_onset_count += 1
                else:
                    # Post-onset delay
                    lead_sec = -abs(float(t_trigger - t_onset))
                    post_onset_count += 1
            else:
                if trigger_idx < onset_idx:
                    lead_sec = float(onset_idx - trigger_idx) * window_seconds
                    pre_onset_count += 1
                elif trigger_idx == onset_idx:
                    lead_sec = 0.0
                    exact_onset_count += 1
                else:
                    lead_sec = -float(trigger_idx - onset_idx) * window_seconds
                    post_onset_count += 1

            lead_times_seconds.append(lead_sec)

    if not lead_times_seconds:
        return {
            "mean_lead_time_seconds": 0.0,
            "median_lead_time_seconds": 0.0,
            "max_lead_time_seconds": 0.0,
            "episodes_detected": 0,
            "pre_onset_warnings": 0,
            "exact_onset_detections": 0,
            "post_onset_detections": 0,
            "total_episodes": len(episodes)
        }

    return {
        "mean_lead_time_seconds": round(float(np.mean(lead_times_seconds)), 2),
        "median_lead_time_seconds": round(float(np.median(lead_times_seconds)), 2),
        "max_lead_time_seconds": round(float(np.max(lead_times_seconds)), 2),
        "episodes_detected": detected_count,
        "pre_onset_warnings": pre_onset_count,
        "exact_onset_detections": exact_onset_count,
        "post_onset_detections": post_onset_count,
        "total_episodes": len(episodes)
    }
