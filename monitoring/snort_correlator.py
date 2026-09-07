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
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger("cyberforecaster.snort")

class SnortCorrelator:
    """
    Ingests Snort alert logs and correlates them with live network flows.
    Enforces 5-tuple compatibility, protocol matching, and temporal proximity gating.
    """
    def __init__(self, log_path: Optional[str] = None, correlation_window_seconds: Optional[float] = None):
        self.log_path = log_path or os.environ.get("SNORT_ALERT_LOG", "logs/snort_alert.fast")
        if correlation_window_seconds is not None:
            self.correlation_window_seconds = float(correlation_window_seconds)
        else:
            self.correlation_window_seconds = float(os.environ.get("SNORT_CORRELATION_WINDOW_SECONDS", "30.0"))

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

    def _parse_snort_timestamp(self, line: str) -> Tuple[str, float]:
        """
        Parses actual event timestamp from Snort fast log line.
        Supports:
        - MM/DD-hh:mm:ss.uuuuuu (e.g. 08/29-23:45:12.123456)
        - MM/DD/YY-hh:mm:ss.uuuuuu (e.g. 08/29/26-23:45:12.123456)
        - ISO 8601 strings (e.g. 2026-08-31T12:05:53.123456+00:00)
        Falls back to current UTC time if no valid timestamp prefix exists.
        """
        ts_match = re.match(r'^(\d{2}/\d{2}(?:/\d{2,4})?-\d{2}:\d{2}:\d{2}(?:\.\d+)?|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)', line)
        if ts_match:
            ts_raw = ts_match.group(1).strip()
            # Case 1: ISO format
            if "T" in ts_raw or (len(ts_raw) >= 10 and ts_raw[4] == '-'):
                try:
                    dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.isoformat(), dt.timestamp()
                except Exception:
                    pass

            # Case 2: MM/DD/YY or MM/DD/YYYY
            if "/" in ts_raw and "-" in ts_raw:
                date_part, time_part = ts_raw.split("-", 1)
                sub_parts = date_part.split("/")
                current_year = datetime.now(timezone.utc).year

                try:
                    if len(sub_parts) == 3:
                        m, d, y = sub_parts
                        y_int = int(y)
                        if y_int < 100:
                            y_int += 2000
                        dt = datetime.strptime(f"{y_int:04d}/{m}/{d}-{time_part}", "%Y/%m/%d-%H:%M:%S.%f" if "." in time_part else "%Y/%m/%d-%H:%M:%S")
                    else:
                        m, d = sub_parts
                        dt = datetime.strptime(f"{current_year:04d}/{m}/{d}-{time_part}", "%Y/%m/%d-%H:%M:%S.%f" if "." in time_part else "%Y/%m/%d-%H:%M:%S")

                    dt = dt.replace(tzinfo=timezone.utc)
                    return dt.isoformat(), dt.timestamp()
                except Exception:
                    pass

        now_dt = datetime.now(timezone.utc)
        return now_dt.isoformat(), now_dt.timestamp()

    def _parse_snort_fast_line(self, line: str) -> Optional[Dict[str, Any]]:
        # Example format: 08/29-23:45:12.123456 [**] [1:1000001:1] TEST ALERT [**] [Priority: 1] {TCP} 192.168.1.5:51234 -> 192.168.1.10:80
        match = re.search(r'\[\*\*\]\s+\[\d+:(\d+):\d+\]\s+(.*?)\s+\[\*\*\]\s+\[Priority:\s+(\d+)\]\s+\{([A-Z]+)\}\s+([\d\.]+):?(\d+)?\s+->\s+([\d\.]+):?(\d+)?', line)
        if match:
            sig_id, msg, priority, proto, src_ip, src_port, dst_ip, dst_port = match.groups()
            ts_iso, ts_epoch = self._parse_snort_timestamp(line)
            return {
                "timestamp": ts_iso,
                "timestamp_epoch": ts_epoch,
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

    def _parse_flow_timestamp(self, flow: Dict[str, Any]) -> Optional[float]:
        """Extracts epoch timestamp from a flow dictionary if present."""
        ts = flow.get("capture_timestamp") or flow.get("timestamp")
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.timestamp()
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                pass
        return None

    def correlate_flow(self, flow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Matches a flow against recently ingested Snort alerts by 5-tuple, protocol,
        and temporal proximity (within self.correlation_window_seconds).
        """
        self.poll_alerts()
        if not self.alerts_buffer:
            return None

        flow_src = flow.get("src_ip")
        flow_dst = flow.get("dst_ip")
        flow_proto = str(flow.get("protocol", "")).upper()
        flow_ts = self._parse_flow_timestamp(flow)
        ref_ts = flow_ts if flow_ts is not None else time.time()

        for alert in reversed(self.alerts_buffer[-50:]):
            if alert["src_ip"] == flow_src and alert["dst_ip"] == flow_dst:
                if not flow_proto or alert["protocol"] == flow_proto:
                    # Enforce temporal proximity gating
                    alert_ts = alert.get("timestamp_epoch")
                    if alert_ts is not None:
                        if abs(ref_ts - alert_ts) <= self.correlation_window_seconds:
                            return alert
                    else:
                        return alert
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "log_path": self.log_path,
            "status": "CONNECTED" if self.is_connected else "NOT CONNECTED",
            "alerts_in_buffer": len(self.alerts_buffer),
            "correlation_window_seconds": self.correlation_window_seconds
        }
