import os
import pandas as pd
import numpy as np
from datetime import datetime

def parse_pcap_file(pcap_path: str) -> pd.DataFrame:
    """
    Parses PCAP file using Scapy or dpkt, extracting flow-level and packet-level features.
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found at: {pcap_path}")

    records = []

    try:
        from scapy.all import rdpcap, IP, TCP, UDP
        packets = rdpcap(pcap_path)

        flow_map = {}

        for pkt in packets:
            if not pkt.haslayer(IP):
                continue

            time_val = float(pkt.time)
            ip = pkt[IP]
            src_ip = ip.src
            dst_ip = ip.dst
            proto = ip.proto
            ttl = ip.ttl
            pkt_len = len(pkt)

            src_port = 0
            dst_port = 0
            flags = ""
            syn = ack = rst = fin = psh = urg = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                flag_str = str(tcp.flags)
                syn = 1 if 'S' in flag_str else 0
                ack = 1 if 'A' in flag_str else 0
                rst = 1 if 'R' in flag_str else 0
                fin = 1 if 'F' in flag_str else 0
                psh = 1 if 'P' in flag_str else 0
                urg = 1 if 'U' in flag_str else 0
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                src_port = udp.sport
                dst_port = udp.dport

            flow_key = (src_ip, dst_ip, src_port, dst_port, proto)

            if flow_key not in flow_map:
                flow_map[flow_key] = {
                    'timestamps': [time_val],
                    'lengths': [pkt_len],
                    'ttls': [ttl],
                    'syn': syn, 'ack': ack, 'rst': rst, 'fin': fin, 'psh': psh, 'urg': urg
                }
            else:
                flow = flow_map[flow_key]
                flow['timestamps'].append(time_val)
                flow['lengths'].append(pkt_len)
                flow['ttls'].append(ttl)
                flow['syn'] += syn
                flow['ack'] += ack
                flow['rst'] += rst
                flow['fin'] += fin
                flow['psh'] += psh
                flow['urg'] += urg

        # Convert aggregated flows to records
        for (s_ip, d_ip, s_port, d_port, proto), f in flow_map.items():
            ts = f['timestamps']
            lens = f['lengths']
            ttls = f['ttls']

            duration = max(0.0001, ts[-1] - ts[0]) if len(ts) > 1 else 0.001
            iats = np.diff(ts) if len(ts) > 1 else np.array([0.0])

            records.append({
                'Timestamp': datetime.fromtimestamp(ts[0]),
                'Src_IP': s_ip,
                'Dst_IP': d_ip,
                'Src_Port': s_port,
                'Dst_Port': d_port,
                'Protocol': proto,
                'Flow_Duration': duration,
                'Tot_Pkts': len(ts),
                'Tot_Bytes': sum(lens),
                'Pkts_Per_Sec': len(ts) / duration,
                'Bytes_Per_Sec': sum(lens) / duration,
                'SYN_Cnt': f['syn'],
                'ACK_Cnt': f['ack'],
                'FIN_Cnt': f['fin'],
                'RST_Cnt': f['rst'],
                'PSH_Cnt': f['psh'],
                'URG_Cnt': f['urg'],
                'Mean_IAT': float(np.mean(iats)),
                'Var_IAT': float(np.var(iats)) if len(iats) > 1 else 0.0,
                'Max_IAT': float(np.max(iats)),
                'Mean_Pkt_Size': float(np.mean(lens)),
                'Var_Pkt_Size': float(np.var(lens)) if len(lens) > 1 else 0.0,
                'TTL_Mean': float(np.mean(ttls)),
                'TTL_Var': float(np.var(ttls)) if len(ttls) > 1 else 0.0,
                'Failed_Conn': 1 if f['rst'] > 0 else 0,
                'Stage': 0,
                'Label': 'Benign'
            })

    except Exception as e:
        print(f"Scapy PCAP parsing fallback error: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
    return df
