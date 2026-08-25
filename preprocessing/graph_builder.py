import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

NODE_FEATURE_DIM = 10
EDGE_FEATURE_DIM = 6

def extract_node_features(df_window: pd.DataFrame, ip_list: List[str]) -> np.ndarray:
    """
    Extracts a 10-dimensional node feature matrix (num_nodes, 10) for a given window.
    Node features:
      0. inbound_packets
      1. outbound_packets
      2. inbound_bytes
      3. outbound_bytes
      4. unique_peers
      5. unique_dst_ports
      6. connection_rate
      7. failed_connection_rate
      8. syn_ratio
      9. ack_ratio
    """
    node_map = {ip: idx for idx, ip in enumerate(ip_list)}
    num_nodes = len(ip_list)
    features = np.zeros((num_nodes, NODE_FEATURE_DIM), dtype=np.float32)

    if df_window.empty:
        return features

    # Pre-group by Src_IP and Dst_IP
    src_grouped = df_window.groupby('Src_IP') if 'Src_IP' in df_window.columns else []
    dst_grouped = df_window.groupby('Dst_IP') if 'Dst_IP' in df_window.columns else []

    for ip, idx in node_map.items():
        in_pkts = out_pkts = 0.0
        in_bytes = out_bytes = 0.0
        peers = set()
        ports = set()
        failed_cnt = 0
        syn_cnt = ack_cnt = total_flw = 0

        # Outbound flows from this IP
        if 'Src_IP' in df_window.columns:
            out_df = df_window[df_window['Src_IP'] == ip]
            if not out_df.empty:
                out_pkts = out_df['Tot_Pkts'].sum() if 'Tot_Pkts' in out_df.columns else len(out_df)
                out_bytes = out_df['Tot_Bytes'].sum() if 'Tot_Bytes' in out_df.columns else 0.0
                peers.update(out_df['Dst_IP'].tolist())
                ports.update(out_df['Dst_Port'].tolist())
                syn_cnt += out_df['SYN_Cnt'].sum() if 'SYN_Cnt' in out_df.columns else 0
                ack_cnt += out_df['ACK_Cnt'].sum() if 'ACK_Cnt' in out_df.columns else 0
                failed_cnt += out_df['Failed_Conn'].sum() if 'Failed_Conn' in out_df.columns else 0
                total_flw += len(out_df)

        # Inbound flows to this IP
        if 'Dst_IP' in df_window.columns:
            in_df = df_window[df_window['Dst_IP'] == ip]
            if not in_df.empty:
                in_pkts = in_df['Tot_Pkts'].sum() if 'Tot_Pkts' in in_df.columns else len(in_df)
                in_bytes = in_df['Tot_Bytes'].sum() if 'Tot_Bytes' in in_df.columns else 0.0
                peers.update(in_df['Src_IP'].tolist())
                total_flw += len(in_df)

        tot_pkts = max(1.0, in_pkts + out_pkts)

        features[idx, 0] = float(in_pkts)
        features[idx, 1] = float(out_pkts)
        features[idx, 2] = float(in_bytes)
        features[idx, 3] = float(out_bytes)
        features[idx, 4] = float(len(peers))
        features[idx, 5] = float(len(ports))
        features[idx, 6] = float(total_flw / 5.0)
        features[idx, 7] = float(failed_cnt / max(1, total_flw))
        features[idx, 8] = float(syn_cnt / tot_pkts)
        features[idx, 9] = float(ack_cnt / tot_pkts)

    return features

def build_window_graph(df_window: pd.DataFrame, window_seconds: float = 5.0) -> Dict:
    """
    Constructs dynamic network graph G(t) = (V(t), E(t)) from a 5-second time window.
    Returns dictionary with nodes, edge index, node features, edge features, and PyG Data.
    """
    if df_window.empty or 'Src_IP' not in df_window.columns or 'Dst_IP' not in df_window.columns:
        # Fallback empty graph with 1 dummy node
        ip_list = ["10.0.0.1"]
        node_features = np.zeros((1, NODE_FEATURE_DIM), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)
    else:
        # Extract unique IP endpoints
        src_ips = df_window['Src_IP'].dropna().unique().tolist()
        dst_ips = df_window['Dst_IP'].dropna().unique().tolist()
        ip_list = sorted(list(set(src_ips + dst_ips)))
        if not ip_list:
            ip_list = ["10.0.0.1"]

        ip_to_idx = {ip: idx for idx, ip in enumerate(ip_list)}

        node_features = extract_node_features(df_window, ip_list)

        # Extract directed communication edges
        edge_src = []
        edge_dst = []
        edge_feat_list = []

        grouped_edges = df_window.groupby(['Src_IP', 'Dst_IP'])
        for (u, v), edge_df in grouped_edges:
            if u in ip_to_idx and v in ip_to_idx:
                edge_src.append(ip_to_idx[u])
                edge_dst.append(ip_to_idx[v])

                flw_cnt = len(edge_df)
                pkts = edge_df['Tot_Pkts'].sum() if 'Tot_Pkts' in edge_df.columns else flw_cnt
                bytes_vol = edge_df['Tot_Bytes'].sum() if 'Tot_Bytes' in edge_df.columns else 0.0
                syn_c = edge_df['SYN_Cnt'].sum() if 'SYN_Cnt' in edge_df.columns else 0
                ack_c = edge_df['ACK_Cnt'].sum() if 'ACK_Cnt' in edge_df.columns else 0
                iat_m = edge_df['Mean_IAT'].mean() if 'Mean_IAT' in edge_df.columns else 0.0

                p_denom = max(1.0, pkts)
                edge_feat_list.append([
                    float(flw_cnt),
                    float(pkts),
                    float(bytes_vol),
                    float(syn_c / p_denom),
                    float(ack_c / p_denom),
                    float(iat_m)
                ])

        if edge_src:
            edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
            edge_features = np.array(edge_feat_list, dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

    # PyTorch Geometric Data object construction
    pyg_data = None
    if HAS_PYG:
        x_t = torch.tensor(node_features, dtype=torch.float32)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        ea_t = torch.tensor(edge_features, dtype=torch.float32)
        pyg_data = PyGData(x=x_t, edge_index=ei_t, edge_attr=ea_t)

    return {
        'ip_list': ip_list,
        'num_nodes': len(ip_list),
        'num_edges': edge_index.shape[1],
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'pyg_data': pyg_data
    }
