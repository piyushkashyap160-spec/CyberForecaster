import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from preprocessing.flow_features import calculate_flow_ratios
from preprocessing.packet_features import calculate_packet_metrics

# Model A — Flow-Only 23-Dimensional State Vector
STATE_FEATURE_KEYS_FLOW_ONLY = [
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

# Model B — Flow + Packet Enriched 30-Dimensional State Vector
STATE_FEATURE_KEYS_ENRICHED = STATE_FEATURE_KEYS_FLOW_ONLY + [
    'pcap_ttl_mean',
    'pcap_ttl_var',
    'pcap_ttl_min',
    'pcap_ttl_max',
    'pcap_pkt_size_var',
    'pcap_iat_var',
    'pcap_port_entropy'
]

# Default alias for backwards compatibility
STATE_FEATURE_KEYS = STATE_FEATURE_KEYS_FLOW_ONLY


def encode_window_to_state(
    df_window: pd.DataFrame,
    window_seconds: float = 5.0,
    pcap_record: Optional[Dict] = None
) -> dict:
    """
    Encodes a single time window dataframe into a 23-dimensional (flow-only) or 30-dimensional (enriched)
    network state dictionary and vector.
    """
    flow_metrics = calculate_flow_ratios(df_window)
    pkt_metrics = calculate_packet_metrics(df_window, window_seconds=window_seconds)

    state_dict = {**flow_metrics, **pkt_metrics}

    if pcap_record:
        # Merge genuine PCAP packet-level metrics
        state_dict['ttl_mean'] = pcap_record.get('pcap_ttl_mean', state_dict['ttl_mean'])
        state_dict['ttl_variance'] = pcap_record.get('pcap_ttl_var', state_dict['ttl_variance'])
        state_dict['pcap_ttl_mean'] = pcap_record.get('pcap_ttl_mean', 64.0)
        state_dict['pcap_ttl_var'] = pcap_record.get('pcap_ttl_var', 0.0)
        state_dict['pcap_ttl_min'] = pcap_record.get('pcap_ttl_min', 64.0)
        state_dict['pcap_ttl_max'] = pcap_record.get('pcap_ttl_max', 64.0)
        state_dict['pcap_pkt_size_var'] = pcap_record.get('pcap_pkt_size_var', state_dict['packet_size_variance'])
        state_dict['pcap_iat_var'] = pcap_record.get('pcap_iat_var', state_dict['IAT_variance'])
        state_dict['pcap_port_entropy'] = pcap_record.get('pcap_port_entropy', state_dict['port_entropy'])
    else:
        # Fallbacks when PCAP is unavailable
        state_dict['pcap_ttl_mean'] = state_dict['ttl_mean']
        state_dict['pcap_ttl_var'] = state_dict['ttl_variance']
        state_dict['pcap_ttl_min'] = state_dict['ttl_mean']
        state_dict['pcap_ttl_max'] = state_dict['ttl_mean']
        state_dict['pcap_pkt_size_var'] = state_dict['packet_size_variance']
        state_dict['pcap_iat_var'] = state_dict['IAT_variance']
        state_dict['pcap_port_entropy'] = state_dict['port_entropy']

    # Vector A: 23-D Flow-only vector
    vector_flow_only = np.array([state_dict[key] for key in STATE_FEATURE_KEYS_FLOW_ONLY], dtype=np.float32)

    # Vector B: 30-D Enriched vector
    vector_enriched = np.array([state_dict[key] for key in STATE_FEATURE_KEYS_ENRICHED], dtype=np.float32)

    # Determine ground truth attack label & stage if present in window
    is_attack = 0
    stage = 0

    if 'Stage' in df_window.columns and not df_window['Stage'].empty:
        stage = int(df_window['Stage'].mode()[0]) if not df_window['Stage'].empty else 0
        is_attack = 1 if stage > 0 else 0
    elif 'Label' in df_window.columns and not df_window['Label'].empty:
        from preprocessing.stage_mapper import map_label_to_stage
        non_benign = [l for l in df_window['Label'].astype(str) if l.lower() != 'benign']
        target_label = non_benign[0] if non_benign else df_window['Label'].iloc[0]
        mapped_info = map_label_to_stage(target_label)
        is_attack = mapped_info['is_attack']
        stage = mapped_info['stage_id']

    return {
        'state_dict': state_dict,
        'vector': vector_flow_only,
        'vector_flow_only': vector_flow_only,
        'vector_enriched': vector_enriched,
        'is_attack': is_attack,
        'stage': stage,
        'timestamp': df_window['Timestamp'].iloc[0] if 'Timestamp' in df_window.columns and not df_window.empty else None
    }
