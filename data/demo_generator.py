import os
import time
import math
import random
import pandas as pd
import numpy as np

def generate_demo_dataset(csv_output_path="data/demo/demo_cicids2018.csv", num_records=4800):
    """
    Generates a realistic multi-scenario flow-level dataset with interleaved benign
    and attack phases across a 1200-second timeline (240 five-second windows).

    Scenario A (0 - 400s): Benign -> Reconnaissance -> Initial Access -> Benign Recovery
    Scenario B (400 - 800s): Benign -> Lateral Movement -> Command & Control -> Benign Recovery
    Scenario C (800 - 1200s): Benign -> Exfiltration -> Benign Recovery

    This multi-scenario timeline guarantees that chronological train (70%), val (15%), and
    test (15%) splits contain realistic mixes of both benign and attack state windows.
    """
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    np.random.seed(42)
    random.seed(42)

    records = []
    base_timestamp = 1700000000.0 # Unix timestamp

    for i in range(num_records):
        # Progress time linearly over 1200 seconds
        rel_time = (i / num_records) * 1200.0
        timestamp = base_timestamp + rel_time

        # Determine stage based on multi-scenario timeline
        if rel_time < 150.0:
            # Scenario A - Phase 1: Benign Baseline
            stage = 0
            label = "Benign"
        elif rel_time < 250.0:
            # Scenario A - Phase 2: Reconnaissance (Port Scan)
            stage = 1
            label = "Reconnaissance"
        elif rel_time < 350.0:
            # Scenario A - Phase 3: Initial Access (Brute Force / Web Attack)
            stage = 2
            label = "Initial Access"
        elif rel_time < 450.0:
            # Scenario A/B - Phase 4: Benign Recovery / Normal Operations
            stage = 0
            label = "Benign"
        elif rel_time < 550.0:
            # Scenario B - Phase 1: Lateral Movement (Internal Probe)
            stage = 3
            label = "Lateral Movement"
        elif rel_time < 650.0:
            # Scenario B - Phase 2: Command & Control (C2 Beaconing)
            stage = 4
            label = "Command & Control"
        elif rel_time < 800.0:
            # Scenario B - Phase 3: Benign Recovery / Normal Operations
            stage = 0
            label = "Benign"
        elif rel_time < 900.0:
            # Scenario C - Phase 1: Benign Baseline
            stage = 0
            label = "Benign"
        elif rel_time < 1050.0:
            # Scenario C - Phase 2: Data Exfiltration
            stage = 5
            label = "Exfiltration"
        else:
            # Scenario C - Phase 3: Benign Recovery
            stage = 0
            label = "Benign"

        # Generate telemetry features with realistic overlap and statistical variance
        if stage == 0: # Benign
            src_ip = f"192.168.1.{random.randint(10, 80)}"
            dst_ip = f"10.0.0.{random.randint(1, 15)}"
            src_port = random.randint(1024, 65535)
            dst_port = random.choice([80, 443, 53, 22, 8080, 445])
            protocol = random.choice([6, 17])
            tot_pkts = random.randint(2, 30)
            tot_bytes = tot_pkts * random.randint(64, 1350)
            syn_cnt = random.choice([0, 1]) if protocol == 6 else 0
            ack_cnt = max(0, tot_pkts - syn_cnt) if protocol == 6 else 0
            fin_cnt = 1 if protocol == 6 and random.random() < 0.2 else 0
            rst_cnt = 1 if protocol == 6 and random.random() < 0.05 else 0
            psh_cnt = random.randint(0, 3)
            urg_cnt = 0
            flow_duration = random.uniform(0.05, 2.5)
            mean_iat = flow_duration / max(1, tot_pkts - 1)
            var_iat = mean_iat * random.uniform(0.1, 0.5)
            max_iat = mean_iat * random.uniform(1.2, 2.5)
            ttl_mean = random.uniform(58, 64)
            ttl_var = random.uniform(0.2, 2.0)
            failed_conn = 1 if rst_cnt > 0 else 0

        elif stage == 1: # Reconnaissance
            src_ip = "192.168.1.100"
            dst_ip = f"10.0.0.{random.randint(1, 10)}"
            src_port = random.randint(30000, 60000)
            dst_port = random.randint(1, 1024)
            protocol = 6
            tot_pkts = random.randint(1, 4)
            tot_bytes = tot_pkts * random.randint(54, 90)
            syn_cnt = tot_pkts
            ack_cnt = random.choice([0, 1])
            fin_cnt = 0
            rst_cnt = random.choice([0, 1])
            psh_cnt = 0
            urg_cnt = 0
            flow_duration = random.uniform(0.001, 0.08)
            mean_iat = flow_duration / max(1, tot_pkts)
            var_iat = max(0.00001, mean_iat * 0.05)
            max_iat = flow_duration
            ttl_mean = random.uniform(52, 56)
            ttl_var = 0.2
            failed_conn = 1 if rst_cnt > 0 else 0

        elif stage == 2: # Initial Access
            src_ip = "192.168.1.100"
            dst_ip = "10.0.0.2"
            src_port = random.randint(40000, 55000)
            dst_port = random.choice([8080, 22, 443])
            protocol = 6
            tot_pkts = random.randint(10, 45)
            tot_bytes = tot_pkts * random.randint(150, 700)
            syn_cnt = random.randint(1, 3)
            ack_cnt = max(0, tot_pkts - syn_cnt - 1)
            fin_cnt = 1
            rst_cnt = random.choice([0, 1, 2])
            psh_cnt = random.randint(2, 8)
            urg_cnt = 0
            flow_duration = random.uniform(0.04, 0.4)
            mean_iat = flow_duration / max(1, tot_pkts)
            var_iat = mean_iat * 0.25
            max_iat = mean_iat * 2.1
            ttl_mean = random.uniform(53, 57)
            ttl_var = 0.3
            failed_conn = random.choice([0, 1])

        elif stage == 3: # Lateral Movement
            src_ip = "10.0.0.2"
            dst_ip = f"10.0.0.{random.randint(3, 12)}"
            src_port = random.randint(49000, 58000)
            dst_port = random.choice([445, 3389, 135, 5985])
            protocol = 6
            tot_pkts = random.randint(20, 70)
            tot_bytes = tot_pkts * random.randint(400, 1100)
            syn_cnt = 1
            ack_cnt = tot_pkts - 2
            fin_cnt = 1
            rst_cnt = random.choice([0, 1])
            psh_cnt = random.randint(4, 12)
            urg_cnt = 0
            flow_duration = random.uniform(0.15, 1.8)
            mean_iat = flow_duration / max(1, tot_pkts)
            var_iat = mean_iat * 0.35
            max_iat = mean_iat * 1.9
            ttl_mean = random.uniform(124, 128)
            ttl_var = 0.4
            failed_conn = 0

        elif stage == 4: # Command & Control
            src_ip = "10.0.0.2"
            dst_ip = "203.0.113.50"
            src_port = random.randint(50000, 62000)
            dst_port = random.choice([443, 80, 53])
            protocol = 6
            tot_pkts = random.randint(8, 22)
            tot_bytes = tot_pkts * random.randint(120, 250)
            syn_cnt = 1
            ack_cnt = tot_pkts - 2
            fin_cnt = 1
            rst_cnt = 0
            psh_cnt = random.randint(2, 6)
            urg_cnt = 0
            flow_duration = random.uniform(0.4, 1.2)
            mean_iat = 0.12 # Regular periodic beaconing
            var_iat = 0.0015 # Low IAT variance
            max_iat = 0.15
            ttl_mean = random.uniform(61, 64)
            ttl_var = 0.1
            failed_conn = 0

        elif stage == 5: # Exfiltration
            src_ip = "10.0.0.2"
            dst_ip = "203.0.113.50"
            src_port = random.randint(55000, 63000)
            dst_port = 443
            protocol = 6
            tot_pkts = random.randint(150, 900)
            tot_bytes = tot_pkts * random.randint(1200, 1460)
            syn_cnt = 1
            ack_cnt = tot_pkts - 2
            fin_cnt = 1
            rst_cnt = 0
            psh_cnt = random.randint(40, 180)
            urg_cnt = 0
            flow_duration = random.uniform(0.8, 3.5)
            mean_iat = flow_duration / max(1, tot_pkts)
            var_iat = mean_iat * 0.08
            max_iat = mean_iat * 1.3
            ttl_mean = random.uniform(61, 64)
            ttl_var = 0.1
            failed_conn = 0

        pkts_per_sec = tot_pkts / max(0.001, flow_duration)
        bytes_per_sec = tot_bytes / max(0.001, flow_duration)
        mean_pkt_size = tot_bytes / max(1, tot_pkts)
        var_pkt_size = mean_pkt_size * random.uniform(0.1, 0.3)

        records.append({
            "Timestamp": pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
            "Src_IP": src_ip,
            "Dst_IP": dst_ip,
            "Src_Port": src_port,
            "Dst_Port": dst_port,
            "Protocol": protocol,
            "Flow_Duration": flow_duration,
            "Tot_Pkts": tot_pkts,
            "Tot_Bytes": tot_bytes,
            "Pkts_Per_Sec": pkts_per_sec,
            "Bytes_Per_Sec": bytes_per_sec,
            "SYN_Cnt": syn_cnt,
            "ACK_Cnt": ack_cnt,
            "FIN_Cnt": fin_cnt,
            "RST_Cnt": rst_cnt,
            "PSH_Cnt": psh_cnt,
            "URG_Cnt": urg_cnt,
            "Mean_IAT": mean_iat,
            "Var_IAT": var_iat,
            "Max_IAT": max_iat,
            "Mean_Pkt_Size": mean_pkt_size,
            "Var_Pkt_Size": var_pkt_size,
            "TTL_Mean": ttl_mean,
            "TTL_Var": ttl_var,
            "Failed_Conn": failed_conn,
            "Stage": stage,
            "Label": label
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)
    print(f"Successfully generated demo CSV dataset with {len(df)} records across 1200s timeline at {csv_output_path}")
    return df

def generate_demo_pcap(pcap_output_path="data/demo/demo_sample.pcap"):
    """
    Generates a simple valid synthetic PCAP file for testing PCAP ingestion parser.
    """
    os.makedirs(os.path.dirname(pcap_output_path), exist_ok=True)
    try:
        from scapy.all import IP, TCP, UDP, Raw, wrpcap
        pkts = []
        base_time = 1700000000.0

        for i in range(150):
            t = base_time + i * 0.1
            ip = IP(src=f"192.168.1.{10 + (i%5)}", dst=f"10.0.0.{1 + (i%2)}")
            if i % 2 == 0:
                l4 = TCP(sport=1024 + i, dport=80, flags="S" if i < 10 else "PA", window=8192)
            else:
                l4 = UDP(sport=2000 + i, dport=53)
            payload = Raw(load=b"X" * (64 + (i * 10) % 500))
            pkt = ip / l4 / payload
            pkt.time = t
            pkts.append(pkt)

        wrpcap(pcap_output_path, pkts)
        print(f"Successfully generated demo PCAP file at {pcap_output_path}")
    except Exception as e:
        print(f"Warning: Could not create demo PCAP using Scapy: {e}")

if __name__ == "__main__":
    generate_demo_dataset()
    generate_demo_pcap()
