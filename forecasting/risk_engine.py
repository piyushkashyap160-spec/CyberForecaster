from typing import List, Dict

STAGE_NAMES = {
    0: "Normal",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command & Control",
    5: "Exfiltration"
}

def calculate_network_risk(rollout_results: List[Dict], warning_threshold: float = 0.70, critical_threshold: float = 0.85) -> Dict:
    """
    Computes aggregated network risk score, maximum future risk, threat level, and active alerts.
    """
    if not rollout_results:
        return {
            'current_risk_score': 0.0,
            'max_future_risk': 0.0,
            'threat_level': 'Normal',
            'alert_triggered': False,
            'predicted_peak_stage': 'Normal'
        }

    probs = [r['attack_probability'] for r in rollout_results]
    stages = [r['predicted_stage_id'] for r in rollout_results]

    current_prob = probs[0]
    max_prob = max(probs)
    peak_stage_id = stages[probs.index(max_prob)]
    peak_stage_name = STAGE_NAMES.get(peak_stage_id, "Normal")

    if max_prob >= critical_threshold:
        threat_level = "CRITICAL"
        alert_triggered = True
    elif max_prob >= warning_threshold:
        threat_level = "HIGH WARNING"
        alert_triggered = True
    elif max_prob >= 0.40:
        threat_level = "ELEVATED"
        alert_triggered = False
    else:
        threat_level = "NORMAL"
        alert_triggered = False

    return {
        'current_risk_score': round(current_prob, 4),
        'max_future_risk': round(max_prob, 4),
        'threat_level': threat_level,
        'alert_triggered': alert_triggered,
        'predicted_peak_stage': peak_stage_name,
        'peak_stage_id': peak_stage_id,
        'rollout_probabilities': probs
    }
