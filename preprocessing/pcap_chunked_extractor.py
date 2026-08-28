"""
pcap_chunked_extractor.py — Scalable Streaming PCAP Packet Feature Extraction Pipeline.

Extracts genuine packet-level telemetry (packet length distributions, IAT statistics,
real IP TTL statistics, TCP flag counts, port entropy) from large PCAP zip archives
in streaming chunks using `dpkt` without loading the full archive into memory.
"""

import os
import io
import math
import zipfile
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Generator
import numpy as np
import pandas as pd
import dpkt

logger = logging.getLogger("cyberforecaster.pcap_extractor")


def calculate_entropy(counts: List[int]) -> float:
    """Calculates Shannon entropy for a list of frequency counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


class ChunkedPCAPExtractor:
    """
    Streaming extractor for raw PCAP archives.
    Parses packets in 5.0-second time windows and extracts 7 key packet-level metrics:
      1. pcap_ttl_mean: Mean IP Time-To-Live
      2. pcap_ttl_var: Variance of IP Time-To-Live
      3. pcap_ttl_min: Minimum IP Time-To-Live
      4. pcap_ttl_max: Maximum IP Time-To-Live
      5. pcap_pkt_size_var: Packet length variance
      6. pcap_iat_var: Inter-arrival time variance
      7. pcap_port_entropy: Destination port Shannon entropy
    """
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = window_seconds

    def extract_from_zip_stream(
        self,
        zip_path: str,
        max_files: Optional[int] = None,
        max_packets_per_file: Optional[int] = 500000
    ) -> List[Dict]:
        """
        Streams PCAP files from zip_path and aggregates packet metrics into 5.0s window dictionary records.
        """
        if not os.path.exists(zip_path):
            logger.warning(f"Zip archive {zip_path} not found.")
            return []

        window_buckets = defaultdict(lambda: {
            "timestamps": [],
            "lengths": [],
            "ttls": [],
            "dst_ports": defaultdict(int),
            "syn_count": 0,
            "ack_count": 0,
            "rst_count": 0,
            "fin_count": 0,
            "tcp_retransmissions": 0,
            "seen_tcp_seqs": set()
        })

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                pcap_names = [n for n in z.namelist() if not n.endswith('/') and ('pcap' in n.lower() or n.endswith('.cap'))]
                if max_files:
                    pcap_names = pcap_names[:max_files]

                logger.info(f"Extracting packet telemetry from {len(pcap_names)} PCAP files in {zip_path}...")

                base_start_ts = None

                for pcap_name in pcap_names:
                    try:
                        with z.open(pcap_name) as f:
                            pcap_reader = dpkt.pcap.Reader(f)
                            pkt_count = 0

                            for ts, buf in pcap_reader:
                                pkt_count += 1
                                if max_packets_per_file and pkt_count > max_packets_per_file:
                                    break

                                if base_start_ts is None:
                                    base_start_ts = ts

                                window_idx = int((ts - base_start_ts) // self.window_seconds)

                                try:
                                    eth = dpkt.ethernet.Ethernet(buf)
                                    if not isinstance(eth.data, dpkt.ip.IP):
                                        continue

                                    ip = eth.data
                                    b_data = window_buckets[window_idx]

                                    b_data["timestamps"].append(ts)
                                    b_data["lengths"].append(len(buf))
                                    b_data["ttls"].append(ip.ttl)

                                    if isinstance(ip.data, dpkt.tcp.TCP):
                                        tcp = ip.data
                                        b_data["dst_ports"][tcp.dport] += 1

                                        if tcp.flags & dpkt.tcp.TH_SYN:
                                            b_data["syn_count"] += 1
                                        if tcp.flags & dpkt.tcp.TH_ACK:
                                            b_data["ack_count"] += 1
                                        if tcp.flags & dpkt.tcp.TH_RST:
                                            b_data["rst_count"] += 1
                                        if tcp.flags & dpkt.tcp.TH_FIN:
                                            b_data["fin_count"] += 1

                                        # Simple retransmission tracking key
                                        seq_key = (ip.src, ip.dst, tcp.sport, tcp.dport, tcp.seq)
                                        if seq_key in b_data["seen_tcp_seqs"]:
                                            b_data["tcp_retransmissions"] += 1
                                        else:
                                            b_data["seen_tcp_seqs"].add(seq_key)

                                    elif isinstance(ip.data, dpkt.udp.UDP):
                                        udp = ip.data
                                        b_data["dst_ports"][udp.dport] += 1

                                except Exception:
                                    continue

                    except Exception as e:
                        logger.warning(f"Skipping malformed pcap entry {pcap_name}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error opening zip archive {zip_path}: {e}")
            return []

        # Convert window buckets into finalized metric records
        records = []
        for win_idx in sorted(window_buckets.keys()):
            data = window_buckets[win_idx]
            ttls = data["ttls"]
            lengths = data["lengths"]
            timestamps = data["timestamps"]

            if not ttls:
                continue

            # Calculate IAT stats
            iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])

            # Port entropy
            port_counts = list(data["dst_ports"].values())
            port_ent = calculate_entropy(port_counts)

            record = {
                "window_idx": win_idx,
                "pcap_packet_count": len(lengths),
                "pcap_pkt_size_mean": round(float(np.mean(lengths)), 2),
                "pcap_pkt_size_var": round(float(np.var(lengths)), 2),
                "pcap_pkt_size_min": float(np.min(lengths)),
                "pcap_pkt_size_max": float(np.max(lengths)),
                "pcap_iat_mean": round(float(np.mean(iats)), 4),
                "pcap_iat_var": round(float(np.var(iats)), 4),
                "pcap_iat_max": round(float(np.max(iats)), 4),
                "pcap_ttl_mean": round(float(np.mean(ttls)), 2),
                "pcap_ttl_var": round(float(np.var(ttls)), 2),
                "pcap_ttl_min": float(np.min(ttls)),
                "pcap_ttl_max": float(np.max(ttls)),
                "pcap_syn_count": data["syn_count"],
                "pcap_ack_count": data["ack_count"],
                "pcap_rst_count": data["rst_count"],
                "pcap_fin_count": data["fin_count"],
                "pcap_tcp_retransmissions": data["tcp_retransmissions"],
                "pcap_port_entropy": port_ent,
                "pcap_enriched_flag": 1.0
            }
            records.append(record)

        logger.info(f"Successfully extracted packet telemetry for {len(records)} 5-second windows.")
        return records
