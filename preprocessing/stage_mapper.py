"""
stage_mapper.py — Transparent Label -> Behavioural Category -> ATT&CK-Aligned Stage Mapper.

This module maps raw flow dataset attack labels (from synthetic generators or real datasets like CIC-IDS2018)
to MITRE ATT&CK-aligned behavioural stages.

Important Scientific Disclaimer:
Network traffic datasets like CIC-IDS2018 do NOT contain native MITRE ATT&CK labels.
This module provides a transparent, expert-defined heuristic mapping layer that maps
dataset attack categories to corresponding ATT&CK tactics based on traffic behaviour.

If a label cannot be defensibly mapped, it is assigned "Unknown / Unmapped" (Stage 0).
"""

from typing import Dict, Tuple

# Stage ID definitions
STAGE_UNKNOWN = 0
STAGE_RECON = 1
STAGE_INITIAL_ACCESS = 2
STAGE_EXECUTION_LATERAL = 3
STAGE_C2 = 4
STAGE_EXFIL_IMPACT = 5

# Comprehensive Stage Metadata Schema
ATTACK_STAGES = {
    0: {
        "id": 0,
        "name": "Benign / Normal / Unmapped",
        "mitre_tactic": "N/A",
        "mitre_id": "N/A",
        "description": "Normal background network traffic or unmapped activity."
    },
    1: {
        "id": 1,
        "name": "Reconnaissance / Probe",
        "mitre_tactic": "Reconnaissance",
        "mitre_id": "TA0043 / T1595",
        "description": "Network scanning, port sweeps, and active probing of IP endpoints."
    },
    2: {
        "id": 2,
        "name": "Initial Access / Brute Force",
        "mitre_tactic": "Initial Access",
        "mitre_id": "TA0001 / T1110 / T1190",
        "description": "Credential guessing, SSH/FTP brute force, or web vulnerability exploitation."
    },
    3: {
        "id": 3,
        "name": "Execution / Lateral Movement",
        "mitre_tactic": "Lateral Movement",
        "mitre_id": "TA0008 / T1021 / T1059",
        "description": "Internal pivot, payload execution, SQL injection, or network infiltration."
    },
    4: {
        "id": 4,
        "name": "Command and Control (C2)",
        "mitre_tactic": "Command and Control",
        "mitre_id": "TA0011 / T1071",
        "description": "Periodic beaconing, botnet traffic, or external C2 server communication."
    },
    5: {
        "id": 5,
        "name": "Exfiltration / Impact (DDoS)",
        "mitre_tactic": "Impact / Exfiltration",
        "mitre_id": "TA0040 / T1498 / TA0010",
        "description": "High-volume data exfiltration, HTTP/UDP flooding, or service disruption."
    }
}

# Configurable Label Mapping Table
# Maps (case-insensitive substring or exact string) -> (Stage ID, Behavioural Rationale, MITRE Tactic ID)
LABEL_MAPPING_RULES: Dict[str, Tuple[int, str, str]] = {
    # Benign
    "benign": (0, "Normal baseline traffic", "N/A"),
    "normal": (0, "Normal baseline traffic", "N/A"),
    
    # Reconnaissance / Scans
    "reconnaissance": (1, "Port scanning and service discovery", "TA0043 / T1595"),
    "portscan": (1, "Active TCP/UDP port sweep", "TA0043 / T1595"),
    "dos attacks-goldeneye": (1, "Application probing and connection starvation", "TA0043 / T1498"),
    "dos attacks-slowloris": (1, "HTTP header starvation probing", "TA0043 / T1498"),
    "dos attacks-slowhttptest": (1, "Low-rate HTTP request starvation probe", "TA0043 / T1498"),
    "dos attacks-hulk": (1, "High-frequency HTTP request probing", "TA0043 / T1498"),
    
    # Initial Access / Brute Force
    "initial_access": (2, "Attempted unauthorized system entry", "TA0001 / T1190"),
    "ftp-bruteforce": (2, "FTP authentication credential guessing", "TA0001 / T1110"),
    "ssh-bruteforce": (2, "SSH authentication credential guessing", "TA0001 / T1110"),
    "brute force -web": (2, "Web application form credential brute-forcing", "TA0001 / T1110"),
    "brute force -xss": (2, "Cross-site scripting injection attempt", "TA0001 / T1190"),
    "sql injection": (2, "Web application SQL database injection", "TA0001 / T1190"),
    
    # Execution & Lateral Movement
    "lateral_movement": (3, "Internal host-to-host movement", "TA0008 / T1021"),
    "infiltration": (3, "Internal network compromise and pivot", "TA0008 / T1021"),
    
    # Command and Control
    "command_and_control": (4, "External beaconing and C2 infrastructure communication", "TA0011 / T1071"),
    "bot": (4, "Botnet client-server C2 traffic", "TA0011 / T1071"),
    
    # Exfiltration & Impact / DDoS
    "exfiltration": (5, "High-volume data transfer to external endpoint", "TA0010 / T1041"),
    "ddos attacks-loic-http": (5, "Volumetric HTTP Distributed Denial of Service", "TA0040 / T1498"),
    "ddos attack-loic-udp": (5, "Volumetric UDP Distributed Denial of Service", "TA0040 / T1498"),
    "ddos attack-hoic": (5, "High-orbit volumetric HTTP flood", "TA0040 / T1498"),
}


def map_label_to_stage(label_str: str) -> Dict:
    """
    Maps a raw flow label string to its ATT&CK-aligned stage details.
    
    Returns dict with:
      - raw_label: str
      - is_attack: int (0 or 1)
      - stage_id: int (0..5)
      - stage_name: str
      - mitre_tactic: str
      - mitre_id: str
      - rationale: str
      - is_defensibly_mapped: bool
    """
    if not isinstance(label_str, str):
        label_str = str(label_str)

    clean_label = label_str.strip().lower()
    
    # Exact or substring match search
    matched_rule = None
    for pattern, rule in LABEL_MAPPING_RULES.items():
        if pattern in clean_label or clean_label in pattern:
            matched_rule = rule
            break

    if matched_rule is not None:
        stage_id, rationale, mitre_id = matched_rule
        stage_info = ATTACK_STAGES.get(stage_id, ATTACK_STAGES[0])
        is_attack = 0 if stage_id == 0 else 1
        return {
            'raw_label': label_str,
            'is_attack': is_attack,
            'stage_id': stage_id,
            'stage_name': stage_info['name'],
            'mitre_tactic': stage_info['mitre_tactic'],
            'mitre_id': mitre_id,
            'rationale': rationale,
            'is_defensibly_mapped': True
        }
    else:
        # Fallback for unmapped / ambiguous labels
        is_attack = 0 if "benign" in clean_label else 1
        return {
            'raw_label': label_str,
            'is_attack': is_attack,
            'stage_id': STAGE_UNKNOWN,
            'stage_name': "Unknown / Unmapped",
            'mitre_tactic': "N/A",
            'mitre_id': "N/A",
            'rationale': "Label could not be defensibly mapped to a specific ATT&CK stage.",
            'is_defensibly_mapped': False
        }
