import os
import time
import math
import random
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Attack Episode Definitions
# ---------------------------------------------------------------------------
# Each episode is a dict:  {start_s, end_s, stage, label, variant}
# 12 episodes spread across 3600s so that 70/15/15 chronological split
# (train: 0–2520s, val: 2520–3060s, test: 3060–3600s)
# guarantees >= 3 episodes in each split.
#
# Stages: 0=Normal, 1=Recon, 2=Initial Access, 3=Lateral Movement,
#         4=C2, 5=Exfiltration
# ---------------------------------------------------------------------------
ATTACK_EPISODES = [
    # --- Train split episodes (0 - 2520s) ---
    {"start":  120, "end":  210, "stage": 1, "label": "Reconnaissance",   "variant": "normal"},
    {"start":  300, "end":  400, "stage": 2, "label": "Initial Access",   "variant": "normal"},
    {"start":  550, "end":  640, "stage": 3, "label": "Lateral Movement", "variant": "normal"},
    {"start":  800, "end":  880, "stage": 4, "label": "Command & Control","variant": "normal"},
    {"start": 1050, "end": 1200, "stage": 5, "label": "Exfiltration",    "variant": "normal"},
    # Slow/stealthy recon: 180s duration, per-flow features look nearly
    # identical to benign — only the sustained slight elevation of SYN ratio
    # accumulated over ~36 consecutive windows reveals the reconnaissance.
    {"start": 1400, "end": 1580, "stage": 1, "label": "Reconnaissance",   "variant": "stealthy"},
    {"start": 1750, "end": 1830, "stage": 2, "label": "Initial Access",   "variant": "normal"},
    {"start": 2100, "end": 2200, "stage": 3, "label": "Lateral Movement", "variant": "normal"},
    # --- Val split episodes (2520 - 3060s) ---
    {"start": 2600, "end": 2700, "stage": 4, "label": "Command & Control","variant": "normal"},
    # --- Test split episodes (3060 - 3600s) ---
    {"start": 3100, "end": 3190, "stage": 1, "label": "Reconnaissance",   "variant": "normal"},
    {"start": 3280, "end": 3370, "stage": 5, "label": "Exfiltration",    "variant": "normal"},
    {"start": 3450, "end": 3540, "stage": 3, "label": "Lateral Movement", "variant": "normal"},
]


def _get_stage(rel_time: float):
    """Return (stage_id, label, episode_index, variant) for a given relative timestamp."""
    for idx, ep in enumerate(ATTACK_EPISODES):
        if ep["start"] <= rel_time < ep["end"]:
            return ep["stage"], ep["label"], idx, ep["variant"]
    return 0, "Benign", -1, "normal"


# ---------------------------------------------------------------------------
# Overlapping IP pools  (benign and attack share subnets)
# ---------------------------------------------------------------------------
_INTERNAL_POOL = [f"192.168.1.{i}" for i in range(10, 120)] + \
                 [f"10.0.0.{i}" for i in range(1, 30)] + \
                 [f"172.16.0.{i}" for i in range(1, 20)]

_EXTERNAL_POOL = [f"203.0.113.{i}" for i in range(1, 60)]


def _pick_src_dst(stage, rng):
    """
    Pick source and destination IPs.  The pools intentionally overlap between
    benign and attack traffic so that IP alone is not perfectly discriminative.
    """
    if stage == 0:
        src = rng.choice(_INTERNAL_POOL)
        dst = rng.choice(_INTERNAL_POOL + _EXTERNAL_POOL[:10])
    elif stage in (1, 2):
        # Recon / Initial Access: internal->internal mostly
        src = rng.choice(_INTERNAL_POOL[:50])     # overlaps benign range
        dst = rng.choice(_INTERNAL_POOL)
    elif stage == 3:
        # Lateral Movement: internal->internal
        src = rng.choice(_INTERNAL_POOL)
        dst = rng.choice(_INTERNAL_POOL)
    elif stage == 4:
        # C2: internal->external
        src = rng.choice(_INTERNAL_POOL[:30])
        dst = rng.choice(_EXTERNAL_POOL)
    elif stage == 5:
        # Exfiltration: internal->external
        src = rng.choice(_INTERNAL_POOL[:30])
        dst = rng.choice(_EXTERNAL_POOL)
    else:
        src = rng.choice(_INTERNAL_POOL)
        dst = rng.choice(_INTERNAL_POOL)
    return src, dst


# ---------------------------------------------------------------------------
# Feature generators per stage with Gaussian noise / jitter
# ---------------------------------------------------------------------------
# KEY DESIGN: benign and attack distributions MUST overlap in individual
# feature dimensions.  The discriminative signal comes from:
#   (a) multi-feature combinations (non-linear boundaries)
#   (b) temporal accumulation over 10+ steps (stealthy recon)
#
# To create overlap:
#   - Benign occasionally has high SYN counts (new connections, retries)
#   - Benign occasionally has low IAT (burst downloads)
#   - Attack features use Gaussian distributions centered slightly away
#     from benign means but with large variance to overlap

def _benign_features(rng):
    """Generate a single benign flow with realistic variance and occasional anomalous-looking features."""
    src_port = rng.randint(1024, 65535)
    dst_port = rng.choice([80, 443, 53, 22, 8080, 445, 8443, 3306, 5432])
    protocol = rng.choice([6, 6, 6, 17])  # 75% TCP, 25% UDP
    tot_pkts = max(1, int(rng.gauss(15, 10)))

    # Occasionally benign traffic has high SYN counts (connection storms, retries)
    if rng.random() < 0.08:
        # Benign connection retry burst — looks like recon to single-step classifier
        syn_cnt = max(1, int(rng.gauss(3, 2))) if protocol == 6 else 0
    else:
        syn_cnt = rng.choice([0, 1]) if protocol == 6 else 0

    tot_bytes = tot_pkts * max(64, int(rng.gauss(500, 300)))
    ack_cnt = max(0, tot_pkts - syn_cnt) if protocol == 6 else 0
    fin_cnt = 1 if protocol == 6 and rng.random() < 0.2 else 0
    rst_cnt = 1 if protocol == 6 and rng.random() < 0.08 else 0
    psh_cnt = rng.randint(0, 5)
    urg_cnt = 0
    flow_duration = max(0.01, rng.gauss(1.0, 0.8))
    mean_iat = flow_duration / max(1, tot_pkts - 1)
    var_iat = abs(mean_iat * rng.gauss(0.3, 0.15))
    max_iat = mean_iat * max(1.0, rng.gauss(1.8, 0.5))
    ttl_mean = max(1, rng.gauss(60, 8))  # wide variance overlaps attack TTL
    ttl_var = abs(rng.gauss(1.0, 0.8))
    failed_conn = 1 if rst_cnt > 0 else 0
    return dict(
        src_port=src_port, dst_port=dst_port, protocol=protocol,
        tot_pkts=tot_pkts, tot_bytes=tot_bytes, syn_cnt=syn_cnt,
        ack_cnt=ack_cnt, fin_cnt=fin_cnt, rst_cnt=rst_cnt, psh_cnt=psh_cnt,
        urg_cnt=urg_cnt, flow_duration=flow_duration, mean_iat=mean_iat,
        var_iat=var_iat, max_iat=max_iat, ttl_mean=ttl_mean, ttl_var=ttl_var,
        failed_conn=failed_conn,
    )


def _recon_features(rng, stealthy=False):
    """
    Reconnaissance flow features.

    Normal recon: elevated SYN ratio, many destination ports, short flows.
    Still uses Gaussian noise so some flows individually overlap with benign.

    Stealthy recon: each flow looks almost identical to benign — normal
    packet count, normal ports, normal duration. The ONLY signal is a
    slightly elevated SYN-to-packet ratio (0.15-0.25 vs benign's ~0.05)
    that only becomes statistically significant when accumulated across
    10+ consecutive 5-second windows. A single-timestep classifier cannot
    reliably detect this.
    """
    if stealthy:
        # Stealthy recon: flows look benign individually
        src_port = rng.randint(1024, 65535)
        dst_port = rng.choice([80, 443, 22, 8080, 53, 445])  # normal ports
        protocol = 6
        tot_pkts = max(2, int(rng.gauss(12, 6)))  # same range as benign
        tot_bytes = tot_pkts * max(64, int(rng.gauss(400, 250)))  # same as benign
        # Key: slightly elevated SYN count — 2 SYN out of ~12 pkts = 0.17 ratio
        # vs benign ~1 SYN out of ~15 pkts = 0.07 ratio
        # Individual flows overlap, but sustained elevation across 30+ windows doesn't
        syn_cnt = max(1, int(rng.gauss(2, 0.8))) if protocol == 6 else 0
        ack_cnt = max(0, tot_pkts - syn_cnt) if protocol == 6 else 0
        fin_cnt = 1 if rng.random() < 0.15 else 0
        rst_cnt = 1 if rng.random() < 0.10 else 0
        psh_cnt = rng.randint(0, 3)
        urg_cnt = 0
        flow_duration = max(0.01, rng.gauss(0.8, 0.5))  # overlaps benign
        ttl_mean = max(1, rng.gauss(59, 6))  # within benign range
    else:
        # Normal (non-stealthy) recon
        src_port = rng.randint(30000, 60000)
        dst_port = rng.randint(1, 1024)
        protocol = 6
        tot_pkts = max(1, int(rng.gauss(4, 2)))
        tot_bytes = tot_pkts * max(54, int(rng.gauss(72, 25)))
        syn_cnt = max(1, tot_pkts)  # All or most packets are SYN
        ack_cnt = rng.choice([0, 1])
        fin_cnt = 0
        rst_cnt = rng.choice([0, 1])
        psh_cnt = 0
        urg_cnt = 0
        flow_duration = max(0.001, rng.gauss(0.05, 0.04))
        ttl_mean = max(1, rng.gauss(55, 6))  # overlaps benign range

    mean_iat = flow_duration / max(1, tot_pkts)
    var_iat = abs(mean_iat * rng.gauss(0.1, 0.05))
    max_iat = flow_duration
    ttl_var = abs(rng.gauss(0.4, 0.2))
    failed_conn = 1 if rst_cnt > 0 else 0
    return dict(
        src_port=src_port, dst_port=dst_port, protocol=protocol,
        tot_pkts=tot_pkts, tot_bytes=tot_bytes, syn_cnt=syn_cnt,
        ack_cnt=ack_cnt, fin_cnt=fin_cnt, rst_cnt=rst_cnt, psh_cnt=psh_cnt,
        urg_cnt=urg_cnt, flow_duration=flow_duration, mean_iat=mean_iat,
        var_iat=var_iat, max_iat=max_iat, ttl_mean=ttl_mean, ttl_var=ttl_var,
        failed_conn=failed_conn,
    )


def _initial_access_features(rng):
    """Initial Access (brute force / web exploit) — moderate overlap with benign."""
    src_port = rng.randint(40000, 55000)
    dst_port = rng.choice([8080, 22, 443, 80, 3389])
    protocol = 6
    tot_pkts = max(5, int(rng.gauss(22, 12)))
    tot_bytes = tot_pkts * max(100, int(rng.gauss(380, 200)))
    syn_cnt = max(1, int(rng.gauss(3, 1.5)))
    ack_cnt = max(0, tot_pkts - syn_cnt - 1)
    fin_cnt = 1
    rst_cnt = rng.choice([0, 1, 2])
    psh_cnt = max(1, int(rng.gauss(5, 3)))
    urg_cnt = 0
    flow_duration = max(0.01, rng.gauss(0.25, 0.2))
    mean_iat = flow_duration / max(1, tot_pkts)
    var_iat = abs(mean_iat * rng.gauss(0.25, 0.12))
    max_iat = mean_iat * max(1.0, rng.gauss(2.0, 0.6))
    ttl_mean = max(1, rng.gauss(56, 6))
    ttl_var = abs(rng.gauss(0.5, 0.25))
    failed_conn = rng.choice([0, 1])
    return dict(
        src_port=src_port, dst_port=dst_port, protocol=protocol,
        tot_pkts=tot_pkts, tot_bytes=tot_bytes, syn_cnt=syn_cnt,
        ack_cnt=ack_cnt, fin_cnt=fin_cnt, rst_cnt=rst_cnt, psh_cnt=psh_cnt,
        urg_cnt=urg_cnt, flow_duration=flow_duration, mean_iat=mean_iat,
        var_iat=var_iat, max_iat=max_iat, ttl_mean=ttl_mean, ttl_var=ttl_var,
        failed_conn=failed_conn,
    )


def _lateral_movement_features(rng):
    """Lateral Movement (internal host hopping) — high TTL separates from most attacks."""
    src_port = rng.randint(49000, 58000)
    dst_port = rng.choice([445, 3389, 135, 5985, 22, 80])
    protocol = 6
    tot_pkts = max(10, int(rng.gauss(40, 18)))
    tot_bytes = tot_pkts * max(200, int(rng.gauss(650, 300)))
    syn_cnt = 1
    ack_cnt = max(0, tot_pkts - 2)
    fin_cnt = 1
    rst_cnt = rng.choice([0, 1])
    psh_cnt = max(2, int(rng.gauss(7, 3)))
    urg_cnt = 0
    flow_duration = max(0.05, rng.gauss(0.9, 0.6))
    mean_iat = flow_duration / max(1, tot_pkts)
    var_iat = abs(mean_iat * rng.gauss(0.35, 0.15))
    max_iat = mean_iat * max(1.0, rng.gauss(1.8, 0.4))
    ttl_mean = max(1, rng.gauss(126, 8))  # internal hops — high TTL
    ttl_var = abs(rng.gauss(0.6, 0.3))
    failed_conn = 0
    return dict(
        src_port=src_port, dst_port=dst_port, protocol=protocol,
        tot_pkts=tot_pkts, tot_bytes=tot_bytes, syn_cnt=syn_cnt,
        ack_cnt=ack_cnt, fin_cnt=fin_cnt, rst_cnt=rst_cnt, psh_cnt=psh_cnt,
        urg_cnt=urg_cnt, flow_duration=flow_duration, mean_iat=mean_iat,
        var_iat=var_iat, max_iat=max_iat, ttl_mean=ttl_mean, ttl_var=ttl_var,
        failed_conn=failed_conn,
    )


def _c2_features(rng):
    """Command & Control (periodic beaconing) — key signal is low IAT variance."""
    src_port = rng.randint(50000, 62000)
    dst_port = rng.choice([443, 80, 53, 8443])
    protocol = 6
    tot_pkts = max(4, int(rng.gauss(14, 6)))
    tot_bytes = tot_pkts * max(80, int(rng.gauss(180, 70)))
    syn_cnt = 1
    ack_cnt = max(0, tot_pkts - 2)
    fin_cnt = 1
    rst_cnt = 0
    psh_cnt = max(1, int(rng.gauss(4, 2)))
    urg_cnt = 0
    flow_duration = max(0.1, rng.gauss(0.8, 0.3))
    mean_iat = max(0.01, rng.gauss(0.12, 0.03))  # periodic beaconing
    var_iat = abs(rng.gauss(0.003, 0.002))  # very low variance
    max_iat = max(mean_iat, mean_iat + rng.gauss(0.03, 0.01))
    ttl_mean = max(1, rng.gauss(62, 5))
    ttl_var = abs(rng.gauss(0.2, 0.1))
    failed_conn = 0
    return dict(
        src_port=src_port, dst_port=dst_port, protocol=protocol,
        tot_pkts=tot_pkts, tot_bytes=tot_bytes, syn_cnt=syn_cnt,
        ack_cnt=ack_cnt, fin_cnt=fin_cnt, rst_cnt=rst_cnt, psh_cnt=psh_cnt,
        urg_cnt=urg_cnt, flow_duration=flow_duration, mean_iat=mean_iat,
        var_iat=var_iat, max_iat=max_iat, ttl_mean=ttl_mean, ttl_var=ttl_var,
        failed_conn=failed_conn,
    )


def _exfiltration_features(rng):
    """Exfiltration (high-volume outbound transfer)."""
    src_port = rng.randint(55000, 63000)
    dst_port = 443
    protocol = 6
    tot_pkts = max(50, int(rng.gauss(350, 250)))
    tot_bytes = tot_pkts * max(800, int(rng.gauss(1250, 250)))
    syn_cnt = 1
    ack_cnt = max(0, tot_pkts - 2)
    fin_cnt = 1
    rst_cnt = 0
    psh_cnt = max(10, int(rng.gauss(70, 50)))
    urg_cnt = 0
    flow_duration = max(0.3, rng.gauss(2.0, 1.0))
    mean_iat = flow_duration / max(1, tot_pkts)
    var_iat = abs(mean_iat * rng.gauss(0.08, 0.03))
    max_iat = mean_iat * max(1.0, rng.gauss(1.3, 0.2))
    ttl_mean = max(1, rng.gauss(62, 5))
    ttl_var = abs(rng.gauss(0.15, 0.08))
    failed_conn = 0
    return dict(
        src_port=src_port, dst_port=dst_port, protocol=protocol,
        tot_pkts=tot_pkts, tot_bytes=tot_bytes, syn_cnt=syn_cnt,
        ack_cnt=ack_cnt, fin_cnt=fin_cnt, rst_cnt=rst_cnt, psh_cnt=psh_cnt,
        urg_cnt=urg_cnt, flow_duration=flow_duration, mean_iat=mean_iat,
        var_iat=var_iat, max_iat=max_iat, ttl_mean=ttl_mean, ttl_var=ttl_var,
        failed_conn=failed_conn,
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_demo_dataset(csv_output_path="data/demo/demo_cicids2018.csv",
                          num_records=14400):
    """
    Generates a hardened multi-episode synthetic flow-level dataset across a
    3600-second timeline (~720 five-second windows, ~14400 flows).

    Key design goals:
      - 12 attack episodes spread across train/val/test splits
      - Feature-level noise: overlapping IP pools, Gaussian-distributed packet
        counts, TTLs, and IATs so benign and attack distributions partially
        overlap at the single-flow level
      - One "slow/stealthy" reconnaissance episode (1400-1580s) where
        individual flows look benign — only sustained SYN ratio elevation
        accumulated over ~30 consecutive timesteps reveals the attack
      - Benign traffic includes occasional "noisy" flows (high SYN, short
        duration) to create genuine overlap with attack distributions
      - At least 3 independent attack episodes in the test split (3060-3600s)
    """
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    rng = random.Random(42)
    np.random.seed(42)

    TOTAL_SECONDS = 3600.0
    records = []
    base_timestamp = 1700000000.0

    for i in range(num_records):
        rel_time = (i / num_records) * TOTAL_SECONDS
        timestamp = base_timestamp + rel_time

        stage, label, ep_idx, variant = _get_stage(rel_time)
        is_stealthy = (variant == "stealthy")

        # Generate IP pair
        src_ip, dst_ip = _pick_src_dst(stage, rng)

        # Generate features per stage
        if stage == 0:
            feat = _benign_features(rng)
        elif stage == 1:
            feat = _recon_features(rng, stealthy=is_stealthy)
        elif stage == 2:
            feat = _initial_access_features(rng)
        elif stage == 3:
            feat = _lateral_movement_features(rng)
        elif stage == 4:
            feat = _c2_features(rng)
        elif stage == 5:
            feat = _exfiltration_features(rng)
        else:
            feat = _benign_features(rng)

        # Derived metrics
        pkts_per_sec = feat['tot_pkts'] / max(0.001, feat['flow_duration'])
        bytes_per_sec = feat['tot_bytes'] / max(0.001, feat['flow_duration'])
        mean_pkt_size = feat['tot_bytes'] / max(1, feat['tot_pkts'])
        var_pkt_size = abs(mean_pkt_size * rng.gauss(0.2, 0.08))

        records.append({
            "Timestamp": pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
            "Src_IP": src_ip,
            "Dst_IP": dst_ip,
            "Src_Port": feat['src_port'],
            "Dst_Port": feat['dst_port'],
            "Protocol": feat['protocol'],
            "Flow_Duration": feat['flow_duration'],
            "Tot_Pkts": feat['tot_pkts'],
            "Tot_Bytes": feat['tot_bytes'],
            "Pkts_Per_Sec": pkts_per_sec,
            "Bytes_Per_Sec": bytes_per_sec,
            "SYN_Cnt": feat['syn_cnt'],
            "ACK_Cnt": feat['ack_cnt'],
            "FIN_Cnt": feat['fin_cnt'],
            "RST_Cnt": feat['rst_cnt'],
            "PSH_Cnt": feat['psh_cnt'],
            "URG_Cnt": feat['urg_cnt'],
            "Mean_IAT": feat['mean_iat'],
            "Var_IAT": feat['var_iat'],
            "Max_IAT": feat['max_iat'],
            "Mean_Pkt_Size": mean_pkt_size,
            "Var_Pkt_Size": var_pkt_size,
            "TTL_Mean": feat['ttl_mean'],
            "TTL_Var": feat['ttl_var'],
            "Failed_Conn": feat['failed_conn'],
            "Stage": stage,
            "Label": label,
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)

    # Validation summary
    total_windows = int(TOTAL_SECONDS / 5)
    attack_flows = len(df[df['Stage'] > 0])
    benign_flows = len(df[df['Stage'] == 0])
    test_start_s = TOTAL_SECONDS * 0.85  # 3060s for 70/15/15 split
    test_episodes = [ep for ep in ATTACK_EPISODES if ep['start'] >= test_start_s]

    print(f"Successfully generated hardened demo CSV dataset:")
    print(f"  Records: {len(df)} | Timeline: {TOTAL_SECONDS:.0f}s | Windows: ~{total_windows}")
    print(f"  Benign flows: {benign_flows} | Attack flows: {attack_flows}")
    print(f"  Attack episodes: {len(ATTACK_EPISODES)} (including 1 stealthy recon)")
    print(f"  Test-split episodes (>= {test_start_s:.0f}s): {len(test_episodes)}")
    for ep in test_episodes:
        print(f"    [{ep['start']}-{ep['end']}s] Stage {ep['stage']}: {ep['label']} ({ep['variant']})")
    print(f"  Output: {csv_output_path}")

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
