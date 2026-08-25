import os
import pandas as pd
import numpy as np
from datetime import datetime

def parse_pcap_file(pcap_path: str) -> pd.DataFrame:
    """
    Parses raw PCAP file using Scapy or dpkt, extracting packet-level and flow-aggregated features.
    
    Packet-Level Derived Features:
      - TTL (mean, variance) from IP header
      - TCP Window Size (mean, min) from TCP header
      - TCP Flags (SYN, ACK, RST, FIN, PSH, URG)
      - IP Fragmentation Flags (DF, MF)
      - Packet Timing & IAT (mean, variance, max)
      - Payload Size Statistics
      - Retransmission Indicators
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
            
            # Fragmentation check
            frag = 1 if (ip.flags & 1 or ip.frag > 0) else 0

            src_port = 0
            dst_port = 0
            tcp_win = 0
            syn = ack = rst = fin = psh = urg = 0
            seq_num = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                tcp_win = getattr(tcp, 'window', 0)
                seq_num = getattr(tcp, 'seq', 0)
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
                    'tcp_wins': [tcp_win] if tcp_win > 0 else [],
                    'seq_nums': [seq_num] if seq_num > 0 else [],
                    'frags': frag,
                    'syn': syn, 'ack': ack, 'rst': rst, 'fin': fin, 'psh': psh, 'urg': urg
                }
            else:
                flow = flow_map[flow_key]
                flow['timestamps'].append(time_val)
                flow['lengths'].append(pkt_len)
                flow['ttls'].append(ttl)
                if tcp_win > 0:
                    flow['tcp_wins'].append(tcp_win)
                if seq_num > 0:
                    flow['seq_nums'].append(seq_num)
                flow['frags'] += frag
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
            tcp_wins = f['tcp_wins']
            seqs = f['seq_nums']

            duration = max(0.0001, ts[-1] - ts[0]) if len(ts) > 1 else 0.001
            iats = np.diff(ts) if len(ts) > 1 else np.array([0.0])
            
            # Simple retransmission estimate: duplicate sequence numbers
            retrans_cnt = max(0, len(seqs) - len(set(seqs))) if len(seqs) > 1 else 0

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
                'TCP_Win_Mean': float(np.mean(tcp_wins)) if tcp_wins else 0.0,
                'Retrans_Cnt': int(retrans_cnt),
                'Fragment_Cnt': int(f['frags']),
                'Failed_Conn': 1 if f['rst'] > 0 else 0,
                'Stage': 0,
                'Label': 'Benign'
            })

    except Exception as e:
        print(f"Scapy PCAP parsing error: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
    return df
