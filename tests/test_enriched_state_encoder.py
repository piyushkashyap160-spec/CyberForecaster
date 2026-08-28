import pytest
import numpy as np
import pandas as pd
from preprocessing.state_encoder import (
    encode_window_to_state,
    STATE_FEATURE_KEYS_FLOW_ONLY,
    STATE_FEATURE_KEYS_ENRICHED
)

def test_state_encoder_schemas():
    assert len(STATE_FEATURE_KEYS_FLOW_ONLY) == 23
    assert len(STATE_FEATURE_KEYS_ENRICHED) == 30

def test_encode_window_to_state_without_pcap():
    df_dummy = pd.DataFrame([{
        'Timestamp': '2018-03-02 08:47:38',
        'Src_IP': '192.168.1.10',
        'Dst_IP': '10.0.0.1',
        'Src_Port': 1024,
        'Dst_Port': 80,
        'Protocol': 6,
        'Flow_Duration': 1.0,
        'Tot_Pkts': 10,
        'Tot_Bytes': 1000,
        'SYN_Cnt': 1,
        'ACK_Cnt': 9,
        'FIN_Cnt': 0,
        'RST_Cnt': 0,
        'Mean_IAT': 0.1,
        'Var_IAT': 0.01,
        'Max_IAT': 0.2,
        'Mean_Pkt_Size': 100.0,
        'Var_Pkt_Size': 50.0,
        'TTL_Mean': 64.0,
        'TTL_Var': 0.0,
        'Failed_Conn': 0,
        'Label': 'Benign'
    }])

    res = encode_window_to_state(df_dummy)
    assert len(res['vector_flow_only']) == 23
    assert len(res['vector_enriched']) == 30
    assert res['is_attack'] == 0
    assert res['stage'] == 0

def test_encode_window_to_state_with_pcap():
    df_dummy = pd.DataFrame([{
        'Timestamp': '2018-03-02 08:47:38',
        'Src_IP': '192.168.1.10',
        'Dst_IP': '10.0.0.1',
        'Src_Port': 1024,
        'Dst_Port': 80,
        'Protocol': 6,
        'Flow_Duration': 1.0,
        'Tot_Pkts': 10,
        'Tot_Bytes': 1000,
        'SYN_Cnt': 1,
        'ACK_Cnt': 9,
        'FIN_Cnt': 0,
        'RST_Cnt': 0,
        'Mean_IAT': 0.1,
        'Var_IAT': 0.01,
        'Max_IAT': 0.2,
        'Mean_Pkt_Size': 100.0,
        'Var_Pkt_Size': 50.0,
        'TTL_Mean': 64.0,
        'TTL_Var': 0.0,
        'Failed_Conn': 0,
        'Label': 'SSH-Bruteforce'
    }])

    pcap_rec = {
        'pcap_ttl_mean': 108.4,
        'pcap_ttl_var': 2750.0,
        'pcap_ttl_min': 32.0,
        'pcap_ttl_max': 255.0,
        'pcap_pkt_size_var': 120.0,
        'pcap_iat_var': 0.05,
        'pcap_port_entropy': 1.58
    }

    res = encode_window_to_state(df_dummy, pcap_record=pcap_rec)
    assert res['state_dict']['ttl_mean'] == 108.4
    assert res['state_dict']['pcap_ttl_var'] == 2750.0
    assert res['is_attack'] == 1
    assert res['stage'] == 2  # Initial Access / Brute Force
