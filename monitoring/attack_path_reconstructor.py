"""
attack_path_reconstructor.py
============================
Real-time Attack Path Reconstruction and Multi-Hop Lateral Progression engine.

Processes completed network flow events, maintains a bounded sliding window
of directed communication edges, evaluates suspicious telemetry indicators
(Snort signatures, FastFlowDetector anomalies, threat stages), and reconstructs
temporal multi-hop attack paths (e.g., Host A -> Host B -> Host C).
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger("cyberforecaster.attack_path")


class AttackPathReconstructor:
    """
    Maintains observed directed edges and reconstructs multi-hop attack paths.
    """
    def __init__(self, window_seconds: float = 60.0, max_edges: int = 200):
        self.window_seconds = window_seconds
        self.max_edges = max_edges
        self.edge_buffer: List[Dict[str, Any]] = []
        self.active_paths: List[Dict[str, Any]] = []

    def is_edge_suspicious(self, event: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determines whether a flow event carries suspicious evidence.
        Priority:
        1. Snort signature alert
        2. FastFlowDetector suspicious classification
        3. Associated high-threat prediction stage / risk
        """
        snort = event.get("snort_alert")
        if snort and isinstance(snort, dict) and snort.get("sig_id"):
            msg = snort.get("message", "Snort signature alert")
            return True, f"Snort: {msg} (SID {snort.get('sig_id')})"

        fast_det = event.get("fast_detection")
        if fast_det and isinstance(fast_det, dict) and fast_det.get("suspicious"):
            conf = fast_det.get("confidence", 0.0)
            lbl = fast_det.get("predicted_label", "Bot / Suspicious")
            return True, f"Detector: {lbl} ({round(conf * 100)}%)"

        threat_level = float(event.get("threatLevel", 0.0))
        predicted_stage = str(event.get("predictedStage", "")).upper()
        if threat_level >= 0.50 and predicted_stage not in ("NORMAL", "BENIGN", ""):
            return True, f"World Model: {predicted_stage} (Threat {round(threat_level * 100)}%)"

        return False, ""

    def add_flow_event(self, event: Dict[str, Any], hosts_db: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ingests a flow event, records directed edges, and updates reconstructed paths.
        """
        src = str(event.get("hostIp") or event.get("src_ip", "")).strip()
        dst = str(event.get("dstIp") or event.get("dst_ip", "")).strip()

        # Reject empty or self-loop edges
        if not src or not dst or src == dst:
            return self.active_paths

        # Determine timestamp
        raw_ts = event.get("timestamp") or event.get("capture_timestamp")
        edge_time = time.time()
        if raw_ts and isinstance(raw_ts, str):
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                edge_time = dt.timestamp()
            except Exception:
                edge_time = time.time()

        # Check host threat stage if hosts_db provided
        if hosts_db and src in hosts_db:
            host_info = hosts_db[src]
            if "threatLevel" not in event:
                event["threatLevel"] = host_info.get("threatLevel", 0.0)
            if "predictedStage" not in event:
                event["predictedStage"] = host_info.get("predictedStage", "Normal")

        is_suspicious, reason = self.is_edge_suspicious(event)

        edge = {
            "src": src,
            "dst": dst,
            "protocol": str(event.get("protocol", "TCP")).upper(),
            "bytes": int(event.get("total_bytes", event.get("byte_count", 0))),
            "duration": float(event.get("duration", 0.0)),
            "timestamp": edge_time,
            "iso_time": datetime.fromtimestamp(edge_time, tz=timezone.utc).isoformat(),
            "is_suspicious": is_suspicious,
            "evidence": reason,
            "snort_alert": event.get("snort_alert"),
            "fast_detection": event.get("fast_detection")
        }

        # Deduplicate identical recent edge (same src, dst, proto within 1s)
        if self.edge_buffer:
            last = self.edge_buffer[-1]
            if (last["src"] == edge["src"] and last["dst"] == edge["dst"] and
                last["protocol"] == edge["protocol"] and abs(last["timestamp"] - edge["timestamp"]) < 1.0):
                # Update evidence if new one is suspicious
                if edge["is_suspicious"] and not last["is_suspicious"]:
                    last["is_suspicious"] = True
                    last["evidence"] = edge["evidence"]
                return self.reconstruct_attack_paths()

        self.edge_buffer.append(edge)
        self._prune_buffer()
        return self.reconstruct_attack_paths()

    def _prune_buffer(self):
        """Prunes edges outside the sliding time window or exceeding max size."""
        if not self.edge_buffer:
            return
        latest_time = max(e["timestamp"] for e in self.edge_buffer)
        self.edge_buffer = [
            e for e in self.edge_buffer
            if (latest_time - e["timestamp"]) <= self.window_seconds
        ]
        if len(self.edge_buffer) > self.max_edges:
            self.edge_buffer = self.edge_buffer[-self.max_edges:]

    def reconstruct_attack_paths(self) -> List[Dict[str, Any]]:
        """
        Reconstructs multi-hop attack paths from the edge buffer.

        Strict False-Positive Protections:
        - Requires at least 2 distinct sequential hops (A -> B -> C).
        - Distinct progression: destination C must NOT equal origin A (eliminates bidirectional TCP/ping bounce).
        - Destination C must NOT equal intermediate B (no self-loops).
        - Chronological order: Hop 2 timestamp >= Hop 1 timestamp.
        - Suspicious evidence required along the chain.
        """
        raw_paths = []
        suspicious_edges = [e for e in self.edge_buffer if e["is_suspicious"]]

        # Find 2-hop or 3-hop chains
        for i, hop1 in enumerate(suspicious_edges):
            a = hop1["src"]
            b = hop1["dst"]
            t1 = hop1["timestamp"]

            for j, hop2 in enumerate(suspicious_edges):
                if i == j:
                    continue
                # B must be the source of the second hop
                if hop2["src"] == b:
                    c = hop2["dst"]
                    t2 = hop2["timestamp"]

                    # Chronological and non-reversal constraints
                    if t2 >= t1 and c != a and c != b and (t2 - t1) <= self.window_seconds:
                        # Check for optional 3rd hop (C -> D)
                        third_hop = None
                        for k, hop3 in enumerate(suspicious_edges):
                            if k != i and k != j and hop3["src"] == c:
                                d = hop3["dst"]
                                t3 = hop3["timestamp"]
                                if t3 >= t2 and d != a and d != b and d != c and (t3 - t1) <= self.window_seconds:
                                    third_hop = hop3
                                    break

                        hops_list = [
                            {
                                "hop_index": 1,
                                "from": a,
                                "to": b,
                                "protocol": hop1["protocol"],
                                "timestamp": hop1["iso_time"],
                                "evidence": hop1["evidence"]
                            },
                            {
                                "hop_index": 2,
                                "from": b,
                                "to": c,
                                "protocol": hop2["protocol"],
                                "timestamp": hop2["iso_time"],
                                "evidence": hop2["evidence"]
                            }
                        ]

                        final_dest = c
                        end_iso = hop2["iso_time"]

                        if third_hop:
                            hops_list.append({
                                "hop_index": 3,
                                "from": c,
                                "to": third_hop["dst"],
                                "protocol": third_hop["protocol"],
                                "timestamp": third_hop["iso_time"],
                                "evidence": third_hop["evidence"]
                            })
                            final_dest = third_hop["dst"]
                            end_iso = third_hop["iso_time"]

                        path_id = f"path-{a}-{final_dest}-{int(t1)}"

                        if not any(p["path_id"] == path_id for p in raw_paths):
                            raw_paths.append({
                                "path_id": path_id,
                                "source": a,
                                "destination": final_dest,
                                "hop_count": len(hops_list),
                                "hops": hops_list,
                                "start_time": hop1["iso_time"],
                                "last_seen": end_iso,
                                "severity": "CRITICAL" if len(hops_list) >= 3 else "HIGH",
                                "mitre_technique": "TA0008 / T1021 (Lateral Movement)"
                            })

        # Filter out sub-paths that are strictly subsumed by a longer path
        final_paths = []
        for p in raw_paths:
            # Check if this path's hops are a strict subset of any longer path
            is_subpath = False
            p_hops_tuples = [(h["from"], h["to"]) for h in p["hops"]]
            for other in raw_paths:
                if other["hop_count"] > p["hop_count"]:
                    other_hops_tuples = [(h["from"], h["to"]) for h in other["hops"]]
                    if all(ht in other_hops_tuples for ht in p_hops_tuples):
                        is_subpath = True
                        break
            if not is_subpath:
                final_paths.append(p)

        self.active_paths = final_paths
        return self.active_paths

    def get_active_paths(self) -> List[Dict[str, Any]]:
        self._prune_buffer()
        return self.active_paths

    def clear(self):
        self.edge_buffer.clear()
        self.active_paths.clear()
