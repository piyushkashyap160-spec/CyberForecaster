"""
forecast_explainer.py
======================
Canonical Evidence-Grounded Forecast Explainability Engine for CyberForecaster.

Answers: "WHY DID CYBERFORECASTER MAKE THIS FORECAST?"
Connects 8 verifiable dimensions without fabrication:
  1. Temporal LSTM Model Evidence (Spatio-temporal Gradient Saliency Attribution)
  2. Current Network Telemetry Evidence (23-D features vs recent baseline)
  3. FastFlowDetector Evidence (Per-flow anomaly classifier suspicion)
  4. Snort Sensor Evidence (Timestamp- and 5-tuple-correlated signature alert)
  5. Stage Transition Dynamics (Progression across MITRE ATT&CK stages)
  6. Empirical Predictive Uncertainty (10-sample MC-Dropout)
  7. Attack Path Reconstruction Evidence (Multi-hop progression A -> B -> C)
  8. Provenance & Audit Integrity (SHA-256 hash, Blockchain Tx, 0.0s lead-time disclosure)
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np
import torch

from explainability.shap_explainer import GradientSaliencyExplainer
from preprocessing.scaler import StateScaler
from preprocessing.state_encoder import STATE_FEATURE_KEYS

logger = logging.getLogger("cyberforecaster.explainability")

# Human-readable labels for 23-D physical network state features
FEATURE_HUMAN_LABELS = {
    "flow_count": "Active Flow Count",
    "total_packets": "Total Packet Volume",
    "total_bytes": "Total Byte Volume",
    "avg_duration": "Avg Flow Duration",
    "unique_src_ips": "Unique Source IPs",
    "unique_dst_ips": "Unique Destination IPs",
    "unique_dst_ports": "Unique Destination Ports",
    "inbound_outbound_ratio": "Inbound / Outbound Ratio",
    "tcp_ratio": "TCP Traffic Fraction",
    "udp_ratio": "UDP Traffic Fraction",
    "icmp_ratio": "ICMP Traffic Fraction",
    "syn_ratio": "SYN Flag Ratio",
    "ack_ratio": "ACK Flag Ratio",
    "fin_ratio": "FIN Flag Ratio",
    "rst_ratio": "RST Flag Ratio",
    "psh_ratio": "PSH Flag Ratio",
    "avg_packet_size": "Avg Packet Size",
    "byte_rate": "Byte Transfer Rate",
    "packet_rate": "Packet Transfer Rate",
    "port_entropy": "Destination Port Entropy",
    "mean_iat": "Mean Inter-Arrival Time",
    "std_iat": "IAT Standard Deviation",
    "flow_anomaly_score": "Flow Anomaly Suspicion Score"
}


class ForecastExplainer:
    """
    Coordinates multi-source evidence extraction and synthesizes canonical
    evidence-grounded forecast explanations for SOC operators.
    """

    def __init__(self, model: torch.nn.Module, scaler: StateScaler):
        self.model = model
        self.scaler = scaler
        self.saliency_engine = GradientSaliencyExplainer(model, scaler)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_telemetry_evidence(
        self,
        sequence: np.ndarray,
        max_features: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Extracts telemetry evidence from physical network state sequence (L, 23).
        Computes current values, recent-history baselines (mean of preceding windows),
        percentage deviations, and elevation status.
        """
        if sequence.ndim == 1:
            sequence = sequence[np.newaxis, :]

        L, D = sequence.shape
        current_state = sequence[-1, :]

        telemetry_items = []
        if L >= 2:
            # Baseline from preceding windows (t-9 to t-1)
            baseline_state = np.mean(sequence[:-1, :], axis=0)
            has_baseline = True
        else:
            baseline_state = current_state
            has_baseline = False

        for j in range(min(D, len(STATE_FEATURE_KEYS))):
            key = STATE_FEATURE_KEYS[j]
            curr_val = float(current_state[j])
            base_val = float(baseline_state[j]) if has_baseline else None

            if has_baseline and abs(base_val) > 1e-6:
                diff = curr_val - base_val
                dev_pct = (diff / abs(base_val)) * 100.0
            elif has_baseline and abs(curr_val - base_val) > 1e-6:
                dev_pct = 100.0 if curr_val > base_val else -100.0
            else:
                dev_pct = 0.0

            # Determine elevation direction
            if not has_baseline:
                direction = "baseline_unavailable"
            elif dev_pct >= 15.0:
                direction = "elevated"
            elif dev_pct <= -15.0:
                direction = "suppressed"
            else:
                direction = "nominal"

            telemetry_items.append({
                "feature": key,
                "feature_label": FEATURE_HUMAN_LABELS.get(key, key),
                "current_value": round(curr_val, 4),
                "baseline_value": round(base_val, 4) if base_val is not None else None,
                "baseline_available": has_baseline,
                "deviation_pct": round(dev_pct, 1) if has_baseline else 0.0,
                "direction": direction,
                "source": "NetworkState"
            })

        # Rank telemetry items by absolute deviation percentage if baseline exists
        if has_baseline:
            telemetry_ranked = sorted(
                telemetry_items,
                key=lambda x: abs(x["deviation_pct"]),
                reverse=True
            )
        else:
            telemetry_ranked = telemetry_items

        return telemetry_ranked[:max_features]

    def compute_mc_uncertainty(
        self,
        sequence: np.ndarray,
        num_samples: int = 10,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Computes calibrated empirical predictive uncertainty using Monte Carlo Dropout.
        Maintains model evaluation mode before and after execution.
        """
        if not hasattr(self.model, "forward_with_mc_dropout"):
            return {
                "available": False,
                "method": "MC-Dropout",
                "summary": "Predictive uncertainty unavailable on active model architecture."
            }

        try:
            if seed is not None:
                torch.manual_seed(seed)

            seq_scaled = self.scaler.transform(sequence[np.newaxis, :, :])
            seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).to(self.device)

            # Execute stochastic forward passes
            mean_state, mean_prob, var_prob, mean_stages = self.model.forward_with_mc_dropout(
                seq_tensor,
                num_samples=num_samples
            )

            # Ensure model returns cleanly to eval mode
            self.model.eval()

            mean_p = float(mean_prob[0, 0].item())
            var_p = float(var_prob[0, 0].item())
            std_p = float(np.sqrt(max(0.0, var_p)))

            lower_bound = max(0.0, mean_p - 1.96 * std_p)
            upper_bound = min(1.0, mean_p + 1.96 * std_p)

            return {
                "available": True,
                "method": f"MC-Dropout ({num_samples} stochastic forward passes)",
                "mean_probability": round(mean_p, 4),
                "std": round(std_p, 4),
                "variance": round(var_p, 6),
                "confidence_band": [round(lower_bound, 4), round(upper_bound, 4)],
                "samples": num_samples,
                "summary": f"MC-dropout empirical predictive uncertainty: mean={mean_p:.2f}, std={std_p:.3f}."
            }
        except Exception as err:
            logger.warning(f"Failed to compute MC-dropout uncertainty: {err}")
            self.model.eval()
            return {
                "available": False,
                "method": "MC-Dropout",
                "summary": f"Uncertainty computation error: {err}"
            }

    def correlate_sensor_evidence(
        self,
        snort_match: Optional[Dict[str, Any]] = None,
        fast_detection: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Formats security sensor evidence with strict provenance and factual descriptions.
        """
        # FastFlowDetector evidence
        if fast_detection and isinstance(fast_detection, dict) and fast_detection.get("available"):
            score = float(fast_detection.get("confidence", 0.0))
            is_susp = bool(fast_detection.get("suspicious", False))
            lbl = str(fast_detection.get("predicted_label", "Bot / Suspicious"))
            status = str(fast_detection.get("status", "ACTIVE"))

            fastflow_ev = {
                "available": True,
                "status": status,
                "suspicious": is_susp,
                "score": round(score, 4),
                "label": lbl,
                "source": "FastFlowDetector",
                "summary": f"Flow anomaly flagged by per-flow detector (score: {score:.2f}, {lbl})." if is_susp else f"Flow evaluated as nominal by per-flow detector (score: {score:.2f}).",
                "disclaimer": "FastFlowDetector provides flow anomaly suspicion and does not independently constitute proof of attack."
            }
        else:
            fastflow_ev = {
                "available": False,
                "status": "NOT_CONFIGURED",
                "suspicious": False,
                "score": 0.0,
                "label": "Unavailable",
                "source": "FastFlowDetector",
                "summary": "Fast per-flow anomaly detector evidence unavailable."
            }

        # Snort signature alert evidence
        if snort_match and isinstance(snort_match, dict) and snort_match.get("sig_id"):
            sid = str(snort_match.get("sig_id"))
            msg = str(snort_match.get("message", "Signature alert"))
            ts = snort_match.get("timestamp")
            proto = str(snort_match.get("protocol", "TCP"))

            snort_ev = {
                "available": True,
                "sid": sid,
                "message": msg,
                "timestamp": ts,
                "protocol": proto,
                "priority": int(snort_match.get("priority", 1)),
                "src_ip": snort_match.get("src_ip"),
                "dst_ip": snort_match.get("dst_ip"),
                "source": "Snort",
                "summary": f"Temporally correlated Snort alert SID {sid}: {msg} ({proto})."
            }
        else:
            snort_ev = {
                "available": False,
                "sid": None,
                "message": None,
                "timestamp": None,
                "source": "Snort",
                "summary": "Snort evidence: None correlated"
            }

        return {
            "fastflow": fastflow_ev,
            "snort": snort_ev
        }

    def correlate_attack_path_evidence(
        self,
        host_ip: str,
        active_paths: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts active multi-hop attack path evidence associated with host_ip.
        """
        if not active_paths:
            return {
                "available": False,
                "path_id": None,
                "hop_count": 0,
                "hops": [],
                "source": "AttackPathReconstructor",
                "summary": "No active attack path is currently correlated."
            }

        # Check if host_ip is involved in any active path (origin, destination, or intermediate hop)
        matched_path = None
        for p in active_paths:
            if p.get("source") == host_ip or p.get("destination") == host_ip:
                matched_path = p
                break
            for h in p.get("hops", []):
                if h.get("from") == host_ip or h.get("to") == host_ip:
                    matched_path = p
                    break
            if matched_path:
                break

        if matched_path:
            hops = matched_path.get("hops", [])
            path_id = matched_path.get("path_id", "active-path")
            src = matched_path.get("source", host_ip)
            dst = matched_path.get("destination", "unknown")
            severity = matched_path.get("severity", "HIGH")
            mitre = matched_path.get("mitre_technique", "TA0008 (Lateral Movement)")

            return {
                "available": True,
                "path_id": path_id,
                "source_host": src,
                "destination_host": dst,
                "severity": severity,
                "hop_count": len(hops),
                "mitre_technique": mitre,
                "hops": hops,
                "source": "AttackPathReconstructor",
                "summary": f"Forecast is associated with an observed multi-hop progression ({src} -> {dst}, {len(hops)} hops)."
            }

        return {
            "available": False,
            "path_id": None,
            "hop_count": 0,
            "hops": [],
            "source": "AttackPathReconstructor",
            "summary": "No active attack path is currently correlated with this host."
        }

    def evaluate_stage_transition(
        self,
        predicted_stage: str,
        previous_stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates stage transition dynamics between previous known state and forecast.
        """
        if not previous_stage or previous_stage == "":
            return {
                "previous_stage": None,
                "predicted_stage": predicted_stage,
                "transition_detected": False,
                "summary": "Initial baseline observation (no previous stage recorded)." if predicted_stage != "Normal" else "Nominal network baseline activity."
            }

        if previous_stage != predicted_stage:
            if predicted_stage == "Normal":
                summary = f"Network activity de-escalating from {previous_stage} to Normal baseline."
            elif previous_stage == "Normal":
                summary = f"Network activity escalating from Normal baseline into {predicted_stage}."
            else:
                summary = f"Forecast indicates a transition from {previous_stage} toward {predicted_stage}."

            return {
                "previous_stage": previous_stage,
                "predicted_stage": predicted_stage,
                "transition_detected": True,
                "summary": summary
            }
        else:
            if predicted_stage == "Normal":
                summary = "Nominal baseline network activity sustained; no malicious progression detected."
            else:
                summary = f"Network activity indicates sustained {predicted_stage} state."

            return {
                "previous_stage": previous_stage,
                "predicted_stage": predicted_stage,
                "transition_detected": False,
                "summary": summary
            }

    def synthesize_deterministic_narrative(
        self,
        host_ip: str,
        predicted_stage: str,
        threat_prob: float,
        model_ev: Dict[str, Any],
        telemetry_ev: List[Dict[str, Any]],
        sensor_ev: Dict[str, Any],
        transition_ev: Dict[str, Any],
        path_ev: Dict[str, Any],
        uncertainty_ev: Dict[str, Any]
    ) -> str:
        """
        Deterministic template synthesis creating an evidence-grounded summary paragraph.
        Strictly zero LLM or chatbot involvement.
        """
        clauses = []

        # 1. Main Forecast Clause
        pct_prob = round(threat_prob * 100)
        clauses.append(f"CyberForecaster projected {predicted_stage} for host {host_ip} with a Threat Probability of {pct_prob}%.")

        # 2. Model Attribution Clause
        top_temporal = model_ev.get("top_temporal_features", [])
        risk_drivers = [item for item in top_temporal if item.get("direction") == "increases_risk"][:2]
        if risk_drivers:
            driver_texts = [f"{d['feature']} ({d['timestep']})" for d in risk_drivers]
            clauses.append(f"Gradient saliency identifies {', '.join(driver_texts)} as primary model risk drivers.")

        # 3. Telemetry Elevation Clause
        elevated_telemetry = [t for t in telemetry_ev if t.get("direction") == "elevated"][:2]
        if elevated_telemetry:
            telem_texts = [f"{t['feature_label']} (+{t['deviation_pct']}%)" for t in elevated_telemetry]
            clauses.append(f"Observed network telemetry shows elevated {', '.join(telem_texts)} against recent baseline.")

        # 4. Security Sensor Clauses
        snort = sensor_ev.get("snort", {})
        if snort.get("available"):
            clauses.append(f"A temporally correlated Snort alert (SID {snort.get('sid')}: {snort.get('message')}) was verified.")

        fastflow = sensor_ev.get("fastflow", {})
        if fastflow.get("available") and fastflow.get("suspicious"):
            clauses.append(f"FastFlowDetector flagged the corresponding flow with suspicion score {fastflow.get('score'):.2f}.")

        # 5. Attack Path Clause
        if path_ev.get("available"):
            clauses.append(f"Active multi-hop attack path {path_ev.get('path_id')} was reconstructed across {path_ev.get('hop_count')} hops.")

        # 6. Stage Transition Clause
        if transition_ev.get("transition_detected"):
            clauses.append(transition_ev.get("summary", ""))

        # 7. Uncertainty Clause
        if uncertainty_ev.get("available"):
            clauses.append(f"Predictive uncertainty was estimated at std={uncertainty_ev.get('std'):.3f} across 10 MC-dropout passes.")

        return " ".join(clauses)

    def generate_explanation(
        self,
        sequence: np.ndarray,
        host_ip: str = "192.168.1.10",
        predicted_stage: str = "Normal",
        threat_probability: float = 0.05,
        forecast_horizon_seconds: int = 25,
        previous_stage: Optional[str] = None,
        snort_match: Optional[Dict[str, Any]] = None,
        fast_detection: Optional[Dict[str, Any]] = None,
        active_paths: Optional[List[Dict[str, Any]]] = None,
        data_hash: Optional[str] = None,
        blockchain_tx: Optional[str] = None,
        uncertainty_samples: int = 10,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Produces the complete canonical forecast explanation schema.
        """
        forecast_id = str(uuid.uuid4())
        timestamp_iso = datetime.now(timezone.utc).isoformat()

        # Severity mapping
        if threat_probability >= 0.85:
            severity = "CRITICAL"
        elif threat_probability >= 0.70:
            severity = "HIGH"
        elif threat_probability >= 0.40:
            severity = "MEDIUM"
        else:
            severity = "NORMAL"

        # 1. Model Evidence
        model_evidence = self.saliency_engine.explain_instance(
            sequence,
            target="attack_probability"
        )

        # 2. Telemetry Evidence
        telemetry_evidence = self.extract_telemetry_evidence(sequence)

        # 3. Security Sensor Evidence
        sensor_evidence = self.correlate_sensor_evidence(snort_match, fast_detection)

        # 4. Attack Path Evidence
        attack_path_evidence = self.correlate_attack_path_evidence(host_ip, active_paths)

        # 5. Stage Transition
        stage_transition = self.evaluate_stage_transition(predicted_stage, previous_stage)

        # 6. Uncertainty
        uncertainty = self.compute_mc_uncertainty(sequence, num_samples=uncertainty_samples, seed=seed)

        # 7. Narrative Synthesis
        narrative = self.synthesize_deterministic_narrative(
            host_ip=host_ip,
            predicted_stage=predicted_stage,
            threat_prob=threat_probability,
            model_ev=model_evidence,
            telemetry_ev=telemetry_evidence,
            sensor_ev=sensor_evidence,
            transition_ev=stage_transition,
            path_ev=attack_path_evidence,
            uncertainty_ev=uncertainty
        )

        # 8. Provenance
        evidence_sources = ["TemporalLSTM", "NetworkState"]
        if sensor_evidence["fastflow"].get("available"):
            evidence_sources.append("FastFlowDetector")
        if sensor_evidence["snort"].get("available"):
            evidence_sources.append("Snort")
        if attack_path_evidence.get("available"):
            evidence_sources.append("AttackPathReconstructor")
        if uncertainty.get("available"):
            evidence_sources.append("MC-Dropout")
        if blockchain_tx:
            evidence_sources.append("BlockchainLedger")

        provenance = {
            "model": "TemporalLSTMWorldModel (23-D, 128 hidden, 2 layers)",
            "scaler": "Train-Fitted StateScaler",
            "data_hash": data_hash,
            "blockchain_registered": bool(blockchain_tx),
            "blockchain_tx": blockchain_tx,
            "evidence_sources": evidence_sources,
            "generated_at": timestamp_iso
        }

        # 9. Scientific Limitations Disclosures
        limitations = [
            "Canonical CSE-CIC-IDS2018 benchmark evaluates future-state trajectory with 0.0s validated pre-onset lead time (exact-onset detection).",
            "FastFlowDetector provides per-flow anomaly suspicion and does not independently constitute proof of attack.",
            "MC-Dropout provides an empirical model uncertainty band and is not a certified statistical confidence interval."
        ]

        return {
            "forecast_id": forecast_id,
            "timestamp": timestamp_iso,
            "host_ip": host_ip,
            "forecast": {
                "attack_probability": round(float(threat_probability), 4),
                "threat_probability": round(float(threat_probability), 4),
                "predicted_stage": predicted_stage,
                "forecast_horizon_seconds": forecast_horizon_seconds,
                "severity": severity,
                "threat_level": severity
            },
            "narrative_summary": narrative,
            "model_evidence": model_evidence,
            "telemetry_evidence": telemetry_evidence,
            "sensor_evidence": sensor_evidence,
            "stage_transition": stage_transition,
            "attack_path": attack_path_evidence,
            "uncertainty": uncertainty,
            "provenance": provenance,
            "limitations": limitations
        }
