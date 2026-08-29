"""
snort_correlator.py
===================
Snort signature alert ingestion and live flow correlation engine.

Monitors local Snort fast alert logs (if configured and running) and correlates
signature-based detection events with live aggregated network flows.
If Snort is not installed/active on the system, gracefully reports NOT CONNECTED.
"""

import os
import re
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("cyberforecaster.snort")

class SnortCorrelator:
    """
    Ingests Snort alert logs and correlates them with live network flows.
    """
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.environ.get("SNORT_ALERT_LOG", "logs/snort_alert.fast")
        self.alerts_buffer: List[Dict[str, Any]] = []
        self.is_connected = False
        self._check_connection()

    def _check_connection(self) -> bool:
        if os.path.exists(self.log_path):
            self.is_connected = True
            logger.info(f"SnortCorrelator connected to alert log at {self.log_path}")
        else:
            self.is_connected = False
            logger.info(f"Snort alert log not found at {self.log_path}. Snort status: NOT CONNECTED.")
        return self.is_connected

    def poll_alerts(self) -> List[Dict[str, Any]]:
        """Reads new alerts from the Snort log if connected."""
        if not self._check_connection():
            return []

        new_alerts = []
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines[-50:]:  # Last 50 alerts
                    parsed = self._parse_snort_fast_line(line.strip())
                    if parsed and parsed not in self.alerts_buffer:
                        self.alerts_buffer.append(parsed)
                        new_alerts.append(parsed)
            if len(self.alerts_buffer) > 200:
                self.alerts_buffer = self.alerts_buffer[-200:]
        except Exception as e:
            logger.error(f"Error reading Snort log {self.log_path}: {e}")
        return new_alerts

    def _parse_snort_fast_line(self, line: str) -> Optional[Dict[str, Any]]:
        # Example format: 08/29-23:45:12.123456 [**] [1:1000001:1] TEST ALERT [**] [Priority: 1] {TCP} 192.168.1.5:51234 -> 192.168.1.10:80
        match = re.search(r'\[\*\*\]\s+\[\d+:(\d+):\d+\]\s+(.*?)\s+\[\*\*\]\s+\[Priority:\s+(\d+)\]\s+\{([A-Z]+)\}\s+([\d\.]+):?(\d+)?\s+->\s+([\d\.]+):?(\d+)?', line)
        if match:
            sig_id, msg, priority, proto, src_ip, src_port, dst_ip, dst_port = match.groups()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sig_id": sig_id,
                "message": msg,
                "priority": int(priority),
                "protocol": proto,
                "src_ip": src_ip,
                "src_port": int(src_port) if src_port else None,
                "dst_ip": dst_ip,
                "dst_port": int(dst_port) if dst_port else None,
                "raw": line
            }
        return None

    def correlate_flow(self, flow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Matches a flow against recently ingested Snort alerts by 5-tuple."""
        if not self.is_connected or not self.alerts_buffer:
            return None

        flow_src = flow.get("src_ip")
        flow_dst = flow.get("dst_ip")
        flow_proto = str(flow.get("protocol", "")).upper()

        for alert in reversed(self.alerts_buffer[-20:]):
            if alert["src_ip"] == flow_src and alert["dst_ip"] == flow_dst:
                if not flow_proto or alert["protocol"] == flow_proto:
                    return alert
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "log_path": self.log_path,
            "status": "CONNECTED" if self.is_connected else "NOT CONNECTED",
            "alerts_in_buffer": len(self.alerts_buffer)
        }
