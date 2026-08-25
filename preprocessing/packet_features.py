import pandas as pd
import numpy as np

def calculate_packet_metrics(df_window: pd.DataFrame, window_seconds: float = 5.0) -> dict:
    """
    Computes packet-level metrics such as TTL distribution, inbound/outbound ratio, and connection rate.
    """
    ttl_mean = df_window['TTL_Mean'].mean() if 'TTL_Mean' in df_window.columns else 64.0
    ttl_var = df_window['TTL_Var'].mean() if 'TTL_Var' in df_window.columns else 0.0

    inbound_count = 0
    outbound_count = 0

    if 'Src_IP' in df_window.columns and 'Dst_IP' in df_window.columns:
        # Simple heuristic: IPs starting with 10. or 192.168. are internal
        for _, row in df_window.iterrows():
            src = str(row['Src_IP'])
            dst = str(row['Dst_IP'])
            is_src_int = src.startswith('10.') or src.startswith('192.168.') or src.startswith('172.16.')
            is_dst_int = dst.startswith('10.') or dst.startswith('192.168.') or dst.startswith('172.16.')
            if is_src_int and not is_dst_int:
                outbound_count += 1
            elif not is_src_int and is_dst_int:
                inbound_count += 1
            else:
                outbound_count += 0.5
                inbound_count += 0.5

    inbound_outbound_ratio = inbound_count / max(1, outbound_count)
    connection_rate = len(df_window) / max(0.1, window_seconds)

    return {
        'ttl_mean': float(ttl_mean),
        'ttl_variance': float(ttl_var),
        'inbound_outbound_ratio': float(inbound_outbound_ratio),
        'connection_rate': float(connection_rate)
    }
