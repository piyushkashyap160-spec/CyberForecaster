import numpy as np
import pandas as pd
import math

def calculate_entropy(series: pd.Series) -> float:
    """
    Calculates Shannon Entropy for a categorical/discrete series (e.g., ports, IPs).
    """
    if series.empty:
        return 0.0
    counts = series.value_counts()
    probs = counts / len(series)
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)

def calculate_flow_ratios(df_window: pd.DataFrame) -> dict:
    """
    Calculates flow aggregate statistics and protocol/flag ratios for a given time window.
    """
    total_pkts = df_window['Tot_Pkts'].sum() if 'Tot_Pkts' in df_window.columns else len(df_window)
    total_bytes = df_window['Tot_Bytes'].sum() if 'Tot_Bytes' in df_window.columns else 0

    unique_src_ips = df_window['Src_IP'].nunique() if 'Src_IP' in df_window.columns else 1
    unique_dst_ips = df_window['Dst_IP'].nunique() if 'Dst_IP' in df_window.columns else 1
    unique_dst_ports = df_window['Dst_Port'].nunique() if 'Dst_Port' in df_window.columns else 1

    tcp_count = (df_window['Protocol'] == 6).sum() if 'Protocol' in df_window.columns else 0
    udp_count = (df_window['Protocol'] == 17).sum() if 'Protocol' in df_window.columns else 0
    num_flows = max(1, len(df_window))

    tcp_ratio = tcp_count / num_flows
    udp_ratio = udp_count / num_flows

    syn_cnt = df_window['SYN_Cnt'].sum() if 'SYN_Cnt' in df_window.columns else 0
    ack_cnt = df_window['ACK_Cnt'].sum() if 'ACK_Cnt' in df_window.columns else 0
    rst_cnt = df_window['RST_Cnt'].sum() if 'RST_Cnt' in df_window.columns else 0
    fin_cnt = df_window['FIN_Cnt'].sum() if 'FIN_Cnt' in df_window.columns else 0

    pkt_denom = max(1, total_pkts)
    syn_ratio = syn_cnt / pkt_denom
    ack_ratio = ack_cnt / pkt_denom
    rst_ratio = rst_cnt / pkt_denom
    fin_ratio = fin_cnt / pkt_denom

    mean_pkt_size = df_window['Mean_Pkt_Size'].mean() if 'Mean_Pkt_Size' in df_window.columns else (total_bytes / pkt_denom)
    var_pkt_size = df_window['Var_Pkt_Size'].mean() if 'Var_Pkt_Size' in df_window.columns else 0.0

    mean_iat = df_window['Mean_IAT'].mean() if 'Mean_IAT' in df_window.columns else 0.0
    var_iat = df_window['Var_IAT'].mean() if 'Var_IAT' in df_window.columns else 0.0
    max_iat = df_window['Max_IAT'].max() if 'Max_IAT' in df_window.columns else 0.0

    retransmission_rate = rst_cnt / num_flows
    failed_conn_rate = df_window['Failed_Conn'].mean() if 'Failed_Conn' in df_window.columns else 0.0
    port_entropy = calculate_entropy(df_window['Dst_Port']) if 'Dst_Port' in df_window.columns else 0.0

    return {
        'total_packets': float(total_pkts),
        'total_bytes': float(total_bytes),
        'unique_src_ips': float(unique_src_ips),
        'unique_dst_ips': float(unique_dst_ips),
        'unique_dst_ports': float(unique_dst_ports),
        'tcp_ratio': float(tcp_ratio),
        'udp_ratio': float(udp_ratio),
        'syn_ratio': float(syn_ratio),
        'ack_ratio': float(ack_ratio),
        'rst_ratio': float(rst_ratio),
        'fin_ratio': float(fin_ratio),
        'mean_packet_size': float(mean_pkt_size),
        'packet_size_variance': float(var_pkt_size),
        'mean_IAT': float(mean_iat),
        'IAT_variance': float(var_iat),
        'max_IAT': float(max_iat),
        'retransmission_rate': float(retransmission_rate),
        'failed_connection_rate': float(failed_conn_rate),
        'port_entropy': float(port_entropy)
    }
