import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from preprocessing.state_encoder import encode_window_to_state, STATE_FEATURE_KEYS

def build_network_states(df: pd.DataFrame, window_seconds: float = 5.0) -> List[Dict]:
    """
    Groups flow traffic into time windows of length `window_seconds` and builds a list of network state dicts.
    """
    if df.empty or 'Timestamp' not in df.columns:
        raise ValueError("DataFrame is empty or missing 'Timestamp' column.")

    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    start_time = df['Timestamp'].iloc[0]
    end_time = df['Timestamp'].iloc[-1]

    total_duration = (end_time - start_time).total_seconds()
    if total_duration <= 0:
        total_duration = window_seconds

    # Resample or group by fixed time bins
    df_indexed = df.set_index('Timestamp')
    rule = f"{int(window_seconds)}s"

    grouped = df_indexed.resample(rule)

    states = []
    for time_bin, group in grouped:
        if group.empty:
            # Empty window padding: zero activity
            empty_dict = {k: 0.0 for k in STATE_FEATURE_KEYS}
            empty_dict['ttl_mean'] = 64.0
            vector = np.array([empty_dict[k] for k in STATE_FEATURE_KEYS], dtype=np.float32)
            states.append({
                'state_dict': empty_dict,
                'vector': vector,
                'is_attack': 0,
                'stage': 0,
                'timestamp': time_bin,
                'df_window': pd.DataFrame()   # empty df for graph builder
            })
        else:
            df_win = group.reset_index()
            state_info = encode_window_to_state(df_win, window_seconds=window_seconds)
            state_info['timestamp'] = time_bin
            state_info['df_window'] = df_win  # store raw flows for graph construction
            states.append(state_info)

    return states

def create_sequences(states: List[Dict], sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Creates rolling window sequences [S(t-L+1)...S(t)] and target Next State S(t+1), attack prob, stage.
    Returns:
        X: (N, L, 23)
        y_state: (N, 23)
        y_attack: (N,)
        y_stage: (N,)
        timestamps: list of target timestamps
    """
    X_list = []
    y_state_list = []
    y_attack_list = []
    y_stage_list = []
    target_timestamps = []

    vectors = [s['vector'] for s in states]
    is_attacks = [s['is_attack'] for s in states]
    stages = [s['stage'] for s in states]
    timestamps = [s['timestamp'] for s in states]

    for i in range(len(states) - sequence_length):
        seq = vectors[i : i + sequence_length]
        next_vec = vectors[i + sequence_length]
        next_attack = is_attacks[i + sequence_length]
        next_stage = stages[i + sequence_length]
        next_ts = timestamps[i + sequence_length]

        X_list.append(seq)
        y_state_list.append(next_vec)
        y_attack_list.append(next_attack)
        y_stage_list.append(next_stage)
        target_timestamps.append(next_ts)

    if not X_list:
        raise ValueError(f"Insufficient states ({len(states)}) to construct sequence length {sequence_length}.")

    X = np.array(X_list, dtype=np.float32)
    y_state = np.array(y_state_list, dtype=np.float32)
    y_attack = np.array(y_attack_list, dtype=np.float32)
    y_stage = np.array(y_stage_list, dtype=np.int64)

    return X, y_state, y_attack, y_stage, target_timestamps
