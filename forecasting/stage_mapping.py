from typing import Dict

STAGE_DESCRIPTIONS = {
    0: {
        "name": "Normal",
        "mitre_id": "TA0000",
        "description": "Standard benign network baseline activity.",
        "key_indicators": ["Low SYN/RST ratio", "Balanced IAT", "Standard port usage"]
    },
    1: {
        "name": "Reconnaissance",
        "mitre_id": "TA0043",
        "description": "Attacker gathering host/port information prior to targeting.",
        "key_indicators": ["High unique destination ports", "Port scanning (SYN sweep)", "Short IAT"]
    },
    2: {
        "name": "Initial Access",
        "mitre_id": "TA0001",
        "description": "Attacker attempting gain entry via brute-force or web exploit.",
        "key_indicators": ["Spike in failed connections", "Repeated HTTP/Auth payloads", "High request rate"]
    },
    3: {
        "name": "Lateral Movement",
        "mitre_id": "TA0008",
        "description": "Pivot from compromised host to internal corporate targets.",
        "key_indicators": ["Internal IP-to-IP connections", "SMB/RDP/WMI port activity", "Host probing"]
    },
    4: {
        "name": "Command & Control",
        "mitre_id": "TA0011",
        "description": "Persistent communication channel established with external attacker server.",
        "key_indicators": ["Periodic beaconing", "Extremely low IAT variance", "Persistent HTTPS/DNS session"]
    },
    5: {
        "name": "Exfiltration",
        "mitre_id": "TA0010",
        "description": "Stolen data being transferred out of network bounds.",
        "key_indicators": ["Massive outbound volume", "Sustained high Bytes/sec", "Continuous PSH/ACK flags"]
    }
}

def get_stage_details(stage_id: int) -> Dict:
    return STAGE_DESCRIPTIONS.get(stage_id, STAGE_DESCRIPTIONS[0])
