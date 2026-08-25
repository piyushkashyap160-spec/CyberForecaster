import numpy as np
import pandas as pd
from typing import Dict, List

def explain_network_graph(graph_dict: Dict) -> Dict:
    """
    Produces graph-level explainability metrics for a dynamic network window graph G(t).
    Identifies high-degree hub nodes, suspicious edge volumes, and peer expansion.
    """
    ip_list = graph_dict.get('ip_list', [])
    node_features = graph_dict.get('node_features', np.zeros((0, 10)))
    edge_index = graph_dict.get('edge_index', np.zeros((2, 0)))
    edge_features = graph_dict.get('edge_features', np.zeros((0, 6)))

    num_nodes = len(ip_list)
    num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

    node_analysis = []
    for idx, ip in enumerate(ip_list):
        if idx < len(node_features):
            in_pkts = float(node_features[idx, 0])
            out_pkts = float(node_features[idx, 1])
            in_bytes = float(node_features[idx, 2])
            out_bytes = float(node_features[idx, 3])
            peers = float(node_features[idx, 4])
            ports = float(node_features[idx, 5])
            syn_r = float(node_features[idx, 8])
            ack_r = float(node_features[idx, 9])

            risk_score = (peers * 0.3) + (ports * 0.3) + (syn_r * 4.0) + (out_bytes / 1e6 * 2.0)

            node_analysis.append({
                'ip': ip,
                'unique_peers': int(peers),
                'unique_ports': int(ports),
                'syn_ratio': round(syn_r, 4),
                'outbound_bytes': round(out_bytes, 2),
                'suspicious_risk_score': round(risk_score, 2),
                'is_suspicious': risk_score > 3.0
            })

    # Sort nodes by risk score descending
    node_analysis = sorted(node_analysis, key=lambda x: x['suspicious_risk_score'], reverse=True)

    return {
        'total_nodes': num_nodes,
        'total_edges': num_edges,
        'high_risk_nodes': [n for n in node_analysis if n['is_suspicious']],
        'top_nodes_by_degree': node_analysis[:5]
    }
