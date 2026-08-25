import numpy as np
import pandas as pd
from preprocessing.flow_features import calculate_flow_ratios
from preprocessing.packet_features import calculate_packet_metrics

STATE_FEATURE_KEYS = [
    'total_packets',
    'total_bytes',
    'unique_src_ips',
    'unique_dst_ips',
    'unique_dst_ports',
    'tcp_ratio',
    'udp_ratio',
    'syn_ratio',
    'ack_ratio',
    'rst_ratio',
    'fin_ratio',
    'mean_packet_size',
    'packet_size_variance',
    'mean_IAT',
    'IAT_variance',
    'max_IAT',
    'retransmission_rate',
    'ttl_mean',
    'ttl_variance',
    'inbound_outbound_ratio',
    'failed_connection_rate',
    'port_entropy',
    'connection_rate'
]

def encode_window_to_state(df_window: pd.DataFrame, window_seconds: float = 5.0) -> dict:
    """
    Encodes a single time window dataframe into a 23-dimensional network state dictionary and vector.
    """
    flow_metrics = calculate_flow_ratios(df_window)
    pkt_metrics = calculate_packet_metrics(df_window, window_seconds=window_seconds)

    state_dict = {**flow_metrics, **pkt_metrics}
    
    # Extract vector in exact feature order
    vector = np.array([state_dict[key] for key in STATE_FEATURE_KEYS], dtype=np.float32)

    # Determine ground truth attack label & stage if present in window
    is_attack = 0
    stage = 0

    if 'Stage' in df_window.columns:
        # Majority or maximum stage in window
        stage = int(df_window['Stage'].mode()[0]) if not df_window['Stage'].empty else 0
        is_attack = 1 if stage > 0 else 0
    elif 'Label' in df_window.columns:
        labels = df_window['Label'].astype(str)
        is_attack = 1 if any(l.lower() != 'benign' for l in labels) else 0

    return {
        'state_dict': state_dict,
        'vector': vector,
        'is_attack': is_attack,
        'stage': stage,
        'timestamp': df_window['Timestamp'].iloc[0] if 'Timestamp' in df_window.columns and not df_window.empty else None
    }
