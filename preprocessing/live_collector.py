"""
live_network_collector.py
=========================
Live Network Traffic Collector for CYCLONE Backend.

Captures real network traffic using Scapy's AsyncSniffer, aggregates
packets into flows by 5-tuple, and feeds them into the existing Cyclone
ML classification and WebSocket pipeline.

Platform-agnostic: works on WSL, Linux, Windows (with Npcap).
Does NOT hardcode any IP addresses or interface names.
"""

import asyncio
import logging
import re
import threading
import time
import uuid
import subprocess
import json
import os
import warnings
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

# Suppress sklearn parallel warning spam
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("cyclone.live_collector")

# ---------------------------------------------------------------------------
# Lazy Scapy import — allows the rest of the backend to start even if
# Scapy is not installed (test-lab mode doesn't need it).
# ---------------------------------------------------------------------------
_scapy_available = False
_scapy_error = None

try:
    from scapy.all import (
        AsyncSniffer,
        get_if_list,
        get_if_addr,
        conf,
        IP,
        IPv6,
        TCP,
        UDP,
        ICMP,
    )
    _scapy_available = True
except ImportError as e:
    _scapy_error = str(e)
    logger.warning(f"[!] Scapy not available: {e}. Live network capture disabled.")
except Exception as e:
    _scapy_error = str(e)
    logger.warning(f"[!] Scapy import error: {e}. Live network capture disabled.")


# ---------------------------------------------------------------------------
# Flow Aggregation — groups packets into flows by 5-tuple
# ---------------------------------------------------------------------------
class FlowAggregator:
    """
    Aggregates raw packets into network flows keyed by 5-tuple:
    (src_ip, dst_ip, src_port, dst_port, protocol).

    A flow is emitted after `flow_timeout` seconds of inactivity.
    """

    def __init__(self, flow_timeout: float = 5.0):
        self.flow_timeout = flow_timeout
        self._active_flows: dict[tuple, dict] = {}
        self._lock = threading.Lock()
        self.debug_log_counter = 0

    def add_packet(self, packet_info: dict) -> Optional[dict]:
        """
        Adds a parsed packet to the flow table.
        Normalizes bidirectional 5-tuples and finalizes flows on timeout, FIN/RST,
        or maximum active duration.
        """
        ep_a = (packet_info["src_ip"], packet_info["src_port"])
        ep_b = (packet_info["dst_ip"], packet_info["dst_port"])
        if ep_a > ep_b:
            canon_src, canon_src_port = ep_b
            canon_dst, canon_dst_port = ep_a
        else:
            canon_src, canon_src_port = ep_a
            canon_dst, canon_dst_port = ep_b

        key = (
            canon_src,
            canon_dst,
            canon_src_port,
            canon_dst_port,
            packet_info["protocol"],
        )
        now = time.time()
        completed_flow = None
        flags = packet_info.get("tcp_flags", set())

        # Diagnostic log for captured packet
        if self.debug_log_counter < 20 or self.debug_log_counter % 200 == 0:
            print(f"[PACKET] src={packet_info['src_ip']}:{packet_info['src_port']} -> dst={packet_info['dst_ip']}:{packet_info['dst_port']} proto={packet_info['protocol']} len={packet_info['length']}", flush=True)
            print(f"[FLOW_KEY] {key}", flush=True)
        self.debug_log_counter += 1

        with self._lock:
            if key in self._active_flows:
                flow = self._active_flows[key]
                flow_duration = now - flow["first_packet_time"]
                flow_idle = now - flow["last_packet_time"]

                # Check if existing flow should be finalized:
                # 1. Inactivity timeout (> flow_timeout)
                # 2. Connection termination (FIN or RST flag)
                # 3. Maximum active flow chunk duration (> 5.0s)
                # 4. Packet threshold (>= 50 packets)
                is_terminated = "F" in flags or "R" in flags
                if flow_idle > self.flow_timeout or is_terminated or flow_duration >= 5.0 or flow["packet_count"] >= 50:
                    flow["packet_count"] += 1
                    flow["byte_count"] += packet_info["length"]
                    flow["last_packet_time"] = now
                    flow["tcp_flags"].update(flags)
                    completed_flow = self._finalize_flow(self._active_flows.pop(key))
                    print(f"[FLOW_FLUSH] key={key} pkts={completed_flow['packet_count']} bytes={completed_flow['byte_count']} dur={completed_flow['duration']:.2f}s reason={'term' if is_terminated else ('dur' if flow_duration>=5.0 else 'idle')}", flush=True)
                else:
                    # Update active flow
                    flow["packet_count"] += 1
                    flow["byte_count"] += packet_info["length"]
                    flow["last_packet_time"] = now
                    flow["tcp_flags"].update(flags)
                    if self.debug_log_counter <= 20:
                        print(f"[FLOW_UPDATE] key={key} pkts={flow['packet_count']} bytes={flow['byte_count']}", flush=True)
            else:
                # New bidirectional 5-tuple flow: initialize
                self._active_flows[key] = self._new_flow(packet_info, now)
                print(f"[FLOW_CREATE] key={key} active_flows={len(self._active_flows)}", flush=True)

        return completed_flow

    def flush_expired(self) -> list[dict]:
        """Returns all flows that have exceeded the inactivity timeout or max duration."""
        now = time.time()
        completed = []
        with self._lock:
            expired_keys = [
                k for k, v in self._active_flows.items()
                if (now - v["last_packet_time"] > self.flow_timeout) or (now - v["first_packet_time"] >= 5.0)
            ]
            for k in expired_keys:
                finalized = self._finalize_flow(self._active_flows.pop(k))
                completed.append(finalized)
                print(f"[FLOW_FLUSH] (Timer) src={finalized['src_ip']} dst={finalized['dst_ip']} pkts={finalized['packet_count']} bytes={finalized['byte_count']} dur={finalized['duration']:.2f}s", flush=True)
        return completed

    def flush_all(self) -> list[dict]:
        """Flushes and returns all active flows (used on stop)."""
        completed = []
        with self._lock:
            for flow in self._active_flows.values():
                completed.append(self._finalize_flow(flow))
            self._active_flows.clear()
        return completed

    @staticmethod
    def _new_flow(packet_info: dict, now: float) -> dict:
        return {
            "src_ip": packet_info["src_ip"],
            "dst_ip": packet_info["dst_ip"],
            "src_port": packet_info["src_port"],
            "dst_port": packet_info["dst_port"],
            "protocol": packet_info["protocol"],
            "packet_count": 1,
            "byte_count": packet_info["length"],
            "first_packet_time": now,
            "last_packet_time": now,
            "capture_timestamp": datetime.now(timezone.utc).isoformat(),
            "tcp_flags": set(packet_info.get("tcp_flags", set())),
        }

    @staticmethod
    def _finalize_flow(flow: dict) -> dict:
        duration = max(0.001, flow["last_packet_time"] - flow["first_packet_time"])
        avg_pkt_size = flow["byte_count"] / max(1, flow["packet_count"])
        tcp_flags = flow["tcp_flags"]

        return {
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "protocol": flow["protocol"],
            "packet_count": flow["packet_count"],
            "byte_count": flow["byte_count"],
            "duration": round(duration, 4),
            "avg_packet_size": round(avg_pkt_size, 2),
            "capture_timestamp": flow["capture_timestamp"],
            "syn_count": 1 if "S" in tcp_flags else 0,
            "ack_count": 1 if "A" in tcp_flags else 0,
            "rst_count": 1 if "R" in tcp_flags else 0,
            "fin_count": 1 if "F" in tcp_flags else 0,
        }


# ---------------------------------------------------------------------------
# Packet Parser — extracts structured info from raw Scapy packets
# ---------------------------------------------------------------------------
def parse_packet(pkt) -> Optional[dict]:
    """
    Extracts structured packet information from a Scapy packet.
    Returns None if the packet doesn't contain an IP or IPv6 layer.
    """
    if not _scapy_available:
        return None

    has_ipv4 = pkt.haslayer(IP)
    has_ipv6 = pkt.haslayer(IPv6)
    
    if not has_ipv4 and not has_ipv6:
        return None

    if has_ipv4:
        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto_num = ip_layer.proto
    else:
        ip_layer = pkt[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto_num = ip_layer.nh

    # Determine protocol name
    protocol = "OTHER"
    if pkt.haslayer(TCP):
        protocol = "TCP"
    elif pkt.haslayer(UDP):
        protocol = "UDP"
    elif pkt.haslayer(ICMP) or proto_num == 58: # 58 is IPv6-ICMP
        protocol = "ICMP"

    info = {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": 0,
        "dst_port": 0,
        "protocol": protocol,
        "length": len(pkt),
        "tcp_flags": set(),
    }

    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        info["src_port"] = tcp.sport
        info["dst_port"] = tcp.dport
        # Extract TCP flags
        flags = tcp.flags
        flag_str = str(flags) if flags else ""
        for f in ["S", "A", "F", "R", "P", "U"]:
            if f in flag_str:
                info["tcp_flags"].add(f)
    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        info["src_port"] = udp.sport
        info["dst_port"] = udp.dport

    return info


# ---------------------------------------------------------------------------
# LiveNetworkCollector — main capture engine
# ---------------------------------------------------------------------------
class LiveNetworkCollector:
    """
    Manages real-time packet capture on a selected network interface.
    Aggregates packets into flows and emits them for the Cyclone pipeline.

    Thread-safe. Designed to be instantiated once and reused.
    """

    def __init__(self, flow_timeout: float = 5.0):
        self.flow_timeout = flow_timeout
        self._sniffer: Optional[object] = None  # AsyncSniffer instance
        self._wsl_proc: Optional[subprocess.Popen] = None  # WSL sniffer subprocess
        self._wsl_thread: Optional[threading.Thread] = None  # WSL stdout reader thread
        self._aggregator = FlowAggregator(flow_timeout=flow_timeout)
        self._flow_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.packet_listeners = []
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Counters
        self.packets_captured = 0
        self.flows_generated = 0
        self.bytes_captured = 0
        self.last_event_timestamp: Optional[str] = None
        self.capture_interface: Optional[str] = None
        self.capture_start_time: Optional[str] = None
        self._errors: list[str] = []
        self.recent_flows = deque(maxlen=300)

    def _discover_wsl_interfaces(self) -> list[dict]:
        """Discovers native network interfaces inside the WSL guest environment."""
        import shutil
        if not shutil.which("wsl"):
            return []
            
        try:
            # Fast check to see if WSL has active/running distributions
            res = subprocess.run(["wsl", "--list", "--running"], capture_output=True, text=True, timeout=0.5)
            if res.returncode != 0:
                return []
        except Exception:
            return []
            
        wsl_interfaces = []
        try:
            res = subprocess.run(["wsl", "ip", "addr"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                current_iface = None
                for line in res.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line[0].isdigit() and ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            iface_name = parts[1].strip()
                            if "@" in iface_name:
                                iface_name = iface_name.split("@")[0]
                            current_iface = {
                                "name": f"wsl:{iface_name}",
                                "display_name": f"WSL: {iface_name}",
                                "description": f"WSL2 Virtual Interface ({iface_name})",
                                "ip": "N/A",
                                "status": "down"
                            }
                            # Check status from flags or line content
                            flags_part = parts[2] if len(parts) >= 3 else line
                            if "UP" in flags_part or "LOWER_UP" in flags_part:
                                current_iface["status"] = "up"
                            wsl_interfaces.append(current_iface)
                    elif line.startswith("inet ") and current_iface:
                        parts = line.split()
                        if len(parts) >= 2:
                            ip = parts[1].split("/")[0]
                            current_iface["ip"] = ip
                            current_iface["status"] = "up"
        except Exception as e:
            logger.warning(f"WSL interface discovery failed or timed out: {e}")
        return wsl_interfaces

    def _discover_windows_interfaces_fallback(self, interfaces, seen_names, seen_displays):
        friendly_map = {}
        try:
            for iface_id, iface_obj in conf.ifaces.items():
                dev_name = getattr(iface_obj, 'name', None) or ''
                dev_description = getattr(iface_obj, 'description', '') or ''
                dev_ip = getattr(iface_obj, 'ip', None) or 'N/A'
                display = dev_name or dev_description or str(iface_id)
                friendly_map[str(iface_id)] = {
                    "display_name": display,
                    "description": dev_description,
                    "ip": dev_ip,
                }
        except Exception:
            pass

        filter_keywords = [
            "-qos packet scheduler",
            "-wfp ",
            "-native wifi filter",
            "-virtual wifi filter",
            "-hyper-v virtual switch extension",
            "-virtual filtering platform",
            "-lightweight filter",
        ]

        if_list = get_if_list()
        for iface_name in if_list:
            friendly = friendly_map.get(iface_name, {})
            display_name = friendly.get("display_name", iface_name)
            description = friendly.get("description", "")

            lower_disp = display_name.lower()
            lower_desc = description.lower()
            if any(k in lower_disp or k in lower_desc for k in filter_keywords):
                continue

            try:
                ip_addr = get_if_addr(iface_name)
            except Exception:
                ip_addr = "N/A"

            if (not ip_addr or ip_addr == "0.0.0.0") and friendly.get("ip") and friendly["ip"] != "N/A":
                ip_addr = friendly["ip"]

            status = "up" if (ip_addr and ip_addr != "0.0.0.0" and ip_addr != "N/A") else "down"

            # APIPA is allowed as fallback, do not override to N/A

            dedup_key = display_name.strip().lower()
            if dedup_key in seen_displays or iface_name in seen_names:
                continue

            seen_displays.add(dedup_key)
            seen_names.add(iface_name)

            logger.info(f"[Windows Interface] display={display_name} | capture={iface_name} | status={status} | ip={ip_addr or 'N/A'}")

            interfaces.append({
                "name": iface_name,
                "display_name": display_name,
                "description": description,
                "ip": ip_addr or "N/A",
                "status": status,
                "capture_available": True
            })

    def discover_interfaces(self) -> list[dict]:
        """
        Discovers available network interfaces across Windows (via PowerShell / Scapy)
        and WSL (via wsl ip addr). Excludes virtual driver filters and deduplicates entries.
        """
        interfaces = []
        seen_names = set()
        seen_displays = set()

        # 1. Discover WSL interfaces
        for w_iface in self._discover_wsl_interfaces():
            if w_iface["name"] not in seen_names:
                seen_names.add(w_iface["name"])
                seen_displays.add(w_iface["display_name"].strip().lower())
                interfaces.append(w_iface)

        # 2. Discover Windows interfaces (if Scapy is available and we are on Windows)
        if _scapy_available and os.name == "nt":
            try:
                powershell_success = False
                adapters = []
                ip_mapping = {}

                # Attempt Windows-native adapter discovery via PowerShell
                try:
                    cmd_adapter = [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "Get-NetAdapter -IncludeHidden | Select-Object Name, InterfaceDescription, Status, InterfaceGuid, ifIndex, MacAddress | ConvertTo-Json"
                    ]
                    res_adapter = subprocess.run(cmd_adapter, capture_output=True, text=True, timeout=3.0, shell=True)
                    if res_adapter.returncode == 0 and res_adapter.stdout.strip():
                        data = json.loads(res_adapter.stdout)
                        adapters = data if isinstance(data, list) else [data]
                        powershell_success = True
                except Exception as e:
                    logger.warning(f"Failed to query Get-NetAdapter via PowerShell: {e}")

                if powershell_success:
                    try:
                        cmd_ip = [
                            "powershell",
                            "-NoProfile",
                            "-NonInteractive",
                            "-Command",
                            "Get-NetIPAddress -AddressFamily IPv4 | Select-Object IPAddress, InterfaceIndex | ConvertTo-Json"
                        ]
                        res_ip = subprocess.run(cmd_ip, capture_output=True, text=True, timeout=3.0, shell=True)
                        if res_ip.returncode == 0 and res_ip.stdout.strip():
                            ip_data = json.loads(res_ip.stdout)
                            ip_list = ip_data if isinstance(ip_data, list) else [ip_data]
                            for ip_obj in ip_list:
                                iface_idx = ip_obj.get("InterfaceIndex")
                                ip_addr = ip_obj.get("IPAddress")
                                if iface_idx is not None and ip_addr:
                                    if iface_idx not in ip_mapping:
                                        ip_mapping[iface_idx] = []
                                    ip_mapping[iface_idx].append(ip_addr)
                    except Exception as e:
                        logger.warning(f"Failed to query Get-NetIPAddress via PowerShell: {e}")

                logger.info(f"[Windows Discovery] PowerShell adapters: {len(adapters)}")

                if powershell_success:
                    # Build GUID to adapter metadata mapping (key is normalized GUID)
                    windows_adapters_map = {}
                    for ad in adapters:
                        guid = ad.get("InterfaceGuid")
                        if guid:
                            norm_guid = guid.strip(" \t\n\r{}").lower().replace("-", "").replace(" ", "")
                            windows_adapters_map[norm_guid] = ad

                    # Retrieve Scapy interfaces
                    scapy_devices = get_if_list()
                    logger.info(f"[Windows Discovery] Scapy/Npcap devices: {len(scapy_devices)}")

                    # Build Scapy GUID to name mapping
                    scapy_guid_map = {}
                    for sc_device in scapy_devices:
                        guid_match = re.search(r'\{([0-9a-fA-F-]+)\}', sc_device)
                        if guid_match:
                            norm_guid = guid_match.group(1).strip(" \t\n\r{}").lower().replace("-", "").replace(" ", "")
                            scapy_guid_map[norm_guid] = sc_device

                    # Track which Windows adapters were mapped
                    mapped_norm_guids = set()

                    # Filter keywords for Scapy devices not mapped to a Windows adapter
                    filter_keywords = [
                        "-qos packet scheduler",
                        "-wfp ",
                        "-native wifi filter",
                        "-virtual wifi filter",
                        "-hyper-v virtual switch extension",
                        "-virtual filtering platform",
                        "-lightweight filter",
                    ]

                    # 1. Process all Scapy devices and match with Windows adapters
                    for sc_device in scapy_devices:
                        guid_match = re.search(r'\{([0-9a-fA-F-]+)\}', sc_device)
                        norm_guid = ""
                        if guid_match:
                            norm_guid = guid_match.group(1).strip(" \t\n\r{}").lower().replace("-", "").replace(" ", "")

                        adapter_metadata = windows_adapters_map.get(norm_guid) if norm_guid else None

                        if adapter_metadata:
                            mapped_norm_guids.add(norm_guid)
                            
                            display_name = adapter_metadata.get("Name", "Unknown Adapter")
                            description = adapter_metadata.get("InterfaceDescription", "")
                            status_str = adapter_metadata.get("Status", "down")
                            status = "up" if status_str.lower() == "up" else "down"
                            if_index = adapter_metadata.get("ifIndex")
                            mac_address = adapter_metadata.get("MacAddress")

                            # Resolve IP
                            best_ip = "N/A"
                            if if_index in ip_mapping:
                                for ip_addr in ip_mapping[if_index]:
                                    if ip_addr and not ip_addr.startswith("169.254."):
                                        best_ip = ip_addr
                                        break
                                if best_ip == "N/A":
                                    for ip_addr in ip_mapping[if_index]:
                                        if ip_addr:
                                            best_ip = ip_addr
                                            break

                            logger.info(f"[Windows Mapping] {display_name} -> {sc_device}")

                            dedup_key = display_name.strip().lower()
                            if dedup_key in seen_displays or sc_device in seen_names:
                                continue
                            seen_displays.add(dedup_key)
                            seen_names.add(sc_device)

                            logger.info(f"[Windows Interface] display={display_name} | capture={sc_device} | status={status} | ip={best_ip}")

                            interfaces.append({
                                "name": sc_device,
                                "display_name": display_name,
                                "description": description,
                                "ip": best_ip,
                                "status": status,
                                "if_index": if_index,
                                "guid": adapter_metadata.get("InterfaceGuid"),
                                "mac_address": mac_address,
                                "capture_available": True
                            })
                        else:
                            # Scapy device not in Windows adapters list
                            lower_device = sc_device.lower()
                            if any(k in lower_device for k in filter_keywords):
                                continue

                            friendly_name = sc_device
                            desc = ""
                            ip_addr = "N/A"
                            try:
                                for iface_id, iface_obj in conf.ifaces.items():
                                    if str(iface_id) == sc_device:
                                        friendly_name = getattr(iface_obj, 'name', sc_device) or sc_device
                                        desc = getattr(iface_obj, 'description', '') or ''
                                        ip_addr = getattr(iface_obj, 'ip', 'N/A') or 'N/A'
                                        break
                            except Exception:
                                pass

                            if ip_addr == "0.0.0.0":
                                        ip_addr = "N/A"

                            status = "up" if (ip_addr and ip_addr != "N/A") else "down"

                            logger.info(f"[Windows Mapping] Unmapped device: {friendly_name} -> {sc_device}")

                            dedup_key = friendly_name.strip().lower()
                            if dedup_key in seen_displays or sc_device in seen_names:
                                continue
                            seen_displays.add(dedup_key)
                            seen_names.add(sc_device)

                            logger.info(f"[Windows Interface] display={friendly_name} | capture={sc_device} | status={status} | ip={ip_addr}")

                            interfaces.append({
                                "name": sc_device,
                                "display_name": friendly_name,
                                "description": desc or f"Scapy Interface ({sc_device})",
                                "ip": ip_addr,
                                "status": status,
                                "capture_available": True
                            })

                    # 2. Add Windows adapters that did NOT map to any Scapy device
                    for norm_guid, adapter_metadata in windows_adapters_map.items():
                        if norm_guid not in mapped_norm_guids:
                            display_name = adapter_metadata.get("Name", "Unknown Adapter")
                            description = adapter_metadata.get("InterfaceDescription", "")
                            status_str = adapter_metadata.get("Status", "down")
                            status = "up" if status_str.lower() == "up" else "down"
                            if_index = adapter_metadata.get("ifIndex")
                            mac_address = adapter_metadata.get("MacAddress")

                            # Resolve IP
                            best_ip = "N/A"
                            if if_index in ip_mapping:
                                for ip_addr in ip_mapping[if_index]:
                                    if ip_addr and not ip_addr.startswith("169.254."):
                                        best_ip = ip_addr
                                        break
                                if best_ip == "N/A":
                                    for ip_addr in ip_mapping[if_index]:
                                        if ip_addr:
                                            best_ip = ip_addr
                                            break

                            logger.info(f"[Windows Mapping] {display_name} -> NO NPF DEVICE")

                            placeholder_name = f"no_npf_{adapter_metadata.get('InterfaceGuid')}"

                            dedup_key = display_name.strip().lower()
                            if dedup_key in seen_displays or placeholder_name in seen_names:
                                continue
                            seen_displays.add(dedup_key)
                            seen_names.add(placeholder_name)

                            logger.info(f"[Windows Interface] display={display_name} | capture=NO NPF DEVICE | status={status} | ip={best_ip}")

                            interfaces.append({
                                "name": placeholder_name,
                                "display_name": display_name,
                                "description": description,
                                "ip": best_ip,
                                "status": status,
                                "if_index": if_index,
                                "guid": adapter_metadata.get("InterfaceGuid"),
                                "mac_address": mac_address,
                                "capture_available": False
                            })
                else:
                    logger.warning("PowerShell discovery failed or was skipped. Falling back to Scapy-only discovery.")
                    self._discover_windows_interfaces_fallback(interfaces, seen_names, seen_displays)

            except Exception as e:
                logger.error(f"Windows interface discovery failed: {e}")
                try:
                    self._discover_windows_interfaces_fallback(interfaces, seen_names, seen_displays)
                except Exception as ex:
                    logger.error(f"Fallback Windows discovery failed: {ex}")

        elif _scapy_available and os.name != "nt":
            # Non-Windows Scapy discovery (e.g. native Linux)
            try:
                for iface_name in get_if_list():
                    if iface_name in seen_names:
                        continue
                    try:
                        ip_addr = get_if_addr(iface_name)
                    except Exception:
                        ip_addr = "N/A"
                    status = "up" if (ip_addr and ip_addr != "0.0.0.0" and ip_addr != "N/A") else "down"
                    
                    seen_names.add(iface_name)
                    interfaces.append({
                        "name": iface_name,
                        "display_name": iface_name,
                        "description": f"Linux Network Interface ({iface_name})",
                        "ip": ip_addr or "N/A",
                        "status": status,
                        "capture_available": True
                    })
            except Exception as e:
                logger.error(f"Linux interface discovery failed: {e}")

        # 3. Handle case where no interfaces are discovered at all
        if not interfaces:
            interfaces.append({
                "name": "unavailable",
                "ip": "N/A",
                "status": "error",
                "display_name": "No interfaces found",
                "description": f"Scapy: {_scapy_error or 'OK'}, WSL: Checked",
                "capture_available": False
            })

        # Sort: adapters with valid IP first, then 'up' interfaces, then WSL, then alphabetically
        interfaces.sort(key=lambda x: (
            0 if (x.get("ip") and x["ip"] != "N/A" and not x["ip"].startswith("169.254.") and not x["ip"].startswith("127.")) else 1,
            0 if x.get("status") == "up" else 1,
            0 if x.get("name", "").startswith("wsl:") else 1,
            x.get("display_name", "")
        ))
        return interfaces

    def _to_wsl_path(self, win_path: str) -> str:
        """Converts a Windows absolute path to a WSL virtual mount path."""
        path = win_path.replace("\\", "/")
        if ":" in path:
            drive, rest = path.split(":", 1)
            path = f"/mnt/{drive.lower()}{rest}"
        return path

    # ------------------------------------------------------------------
    # Capture Lifecycle
    # ------------------------------------------------------------------
    def start_capture(self, interface: str) -> dict:
        """
        Starts packet capture on the specified interface (Windows or WSL target).
        Returns a status dict with success/error information.
        """
        # Validate interface exists
        discovered = self.discover_interfaces()
        selected_iface_info = None
        for info in discovered:
            if info["name"] == interface:
                selected_iface_info = info
                break

        if selected_iface_info is None:
            return {
                "success": False,
                "error": f"Interface '{interface}' not found. Available: {[i['name'] for i in discovered]}",
            }

        if not selected_iface_info.get("capture_available", True):
            return {
                "success": False,
                "error": f"Interface '{selected_iface_info['display_name']}' does not support live capture (no Npcap device mapped).",
            }

        is_wsl = interface.startswith("wsl:")

        if not is_wsl and not _scapy_available:
            error_msg = f"Cannot start capture on Windows: Scapy not available ({_scapy_error})"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}

        with self._lock:
            if self._running:
                if self.capture_interface == interface:
                    logger.info(f"Capture already running on requested interface '{interface}'. Ignoring start request.")
                    return {"success": True, "interface": interface}
                else:
                    logger.info(f"Switching interface from '{self.capture_interface}' to '{interface}'. Stopping current capture first.")
                    self._stop_capture_internal()

            try:
                # Reset counters
                self.packets_captured = 0
                self.flows_generated = 0
                self.bytes_captured = 0
                self.last_event_timestamp = None
                self._errors.clear()
                self.capture_interface = interface
                self.capture_start_time = datetime.now(timezone.utc).isoformat()
                self._stop_event.clear()

                if is_wsl:
                    # WSL guest capture mode
                    wsl_iface_name = interface.split(":", 1)[1]
                    sniffer_local_path = os.path.join(os.path.dirname(__file__), "wsl_sniffer.py")
                    wsl_sniffer_path = self._to_wsl_path(sniffer_local_path)
                    
                    cmd = ["wsl", "-u", "root", "python3", "-u", wsl_sniffer_path, wsl_iface_name]
                    logger.info(f"Launching WSL sniffer subprocess: {' '.join(cmd)}")
                    self._wsl_proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1
                    )
                    
                    # Spawn reader thread
                    self._wsl_thread = threading.Thread(
                        target=self._read_wsl_stdout,
                        args=(self._wsl_proc, wsl_iface_name),
                        daemon=True,
                        name="cyclone-wsl-reader"
                    )
                    self._wsl_thread.start()
                else:
                    # Windows host capture mode
                    display_name = selected_iface_info.get("display_name", interface)
                    logger.info(f"[Windows Capture] Starting Npcap capture on {display_name} -> {interface}")
                    self._sniffer = AsyncSniffer(
                        iface=interface,
                        prn=self._on_packet,
                        store=False,
                        filter="ip or ip6",
                    )
                    self._sniffer.start()

                    # Verify that capture started successfully
                    time.sleep(0.1)
                    if self._sniffer.exception:
                        raise RuntimeError(f"Scapy failed to start capture: {self._sniffer.exception}")
                    if not self._sniffer.thread or not self._sniffer.thread.is_alive():
                        raise RuntimeError("Scapy sniffer thread stopped immediately after start.")

                self._running = True

                # Start background flow flush thread
                self._flush_thread = threading.Thread(
                    target=self._flush_loop,
                    daemon=True,
                    name="cyclone-flow-flusher",
                )
                self._flush_thread.start()

                logger.info(f"[+] Live capture started on interface: {interface}")
                return {"success": True, "interface": interface}

            except PermissionError as e:
                error_msg = (
                    f"Permission denied for capture on '{interface}'. "
                    f"Run with elevated privileges (sudo/admin). Detail: {e}"
                )
                logger.error(error_msg)
                self._running = False
                if self._sniffer:
                    try:
                        self._sniffer.stop()
                    except Exception:
                        pass
                    self._sniffer = None
                return {"success": False, "error": error_msg}
            except Exception as e:
                error_msg = f"Failed to start capture on '{interface}': {e}"
                logger.error(error_msg)
                self._running = False
                if self._sniffer:
                    try:
                        self._sniffer.stop()
                    except Exception:
                        pass
                    self._sniffer = None
                return {"success": False, "error": error_msg}

    def _stop_capture_internal(self):
        """Stops the active packet capture. Internal method, MUST hold self._lock."""
        if not self._running:
            return
        try:
            self._stop_event.set()
            self._running = False

            # Stop WSL sniffer subprocess
            if self._wsl_proc:
                try:
                    self._wsl_proc.terminate()
                    self._wsl_proc.wait(timeout=2.0)
                except Exception as e:
                    logger.warning(f"Error terminating WSL sniffer process: {e}")
                self._wsl_proc = None

            if self._wsl_thread and self._wsl_thread.is_alive():
                self._wsl_thread.join(timeout=2.0)
            self._wsl_thread = None

            # Stop Scapy Windows sniffer
            if self._sniffer:
                try:
                    self._sniffer.stop()
                except Exception as e:
                    logger.warning(f"Sniffer stop warning: {e}")
                self._sniffer = None

            # Wait for flush thread to finish
            if self._flush_thread and self._flush_thread.is_alive():
                self._flush_thread.join(timeout=3.0)
            self._flush_thread = None

            # Flush remaining flows
            remaining = self._aggregator.flush_all()
            for flow in remaining:
                self._emit_flow(flow)

            logger.info(
                f"[+] Live capture stopped internally. "
                f"Total packets: {self.packets_captured}, "
                f"Total flows: {self.flows_generated}"
            )
        except Exception as e:
            logger.error(f"Error in _stop_capture_internal: {e}")

    def stop_capture(self) -> dict:
        """Stops the active packet capture and flushes remaining flows."""
        with self._lock:
            if not self._running:
                return {"success": True, "message": "No active capture to stop."}

            try:
                self._stop_capture_internal()
                return {
                    "success": True,
                    "packets_captured": self.packets_captured,
                    "flows_generated": self.flows_generated,
                }
            except Exception as e:
                error_msg = f"Error stopping capture: {e}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

    def _read_wsl_stdout(self, proc, iface_name):
        """Reads JSON packets printed by the WSL sniffer subprocess in real time."""
        logger.info(f"WSL sniffer reader thread started for interface: {iface_name}")
        while not self._stop_event.is_set() and proc.poll() is None:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                pkt_data = json.loads(line)
                if "error" in pkt_data:
                    logger.error(f"[WSL Error] {pkt_data['error']}")
                    self._errors.append(f"WSL error: {pkt_data['error']}")
                    continue

                self.packets_captured += 1
                self.bytes_captured += pkt_data.get("length", 0)
                self.last_event_timestamp = datetime.now(timezone.utc).isoformat()

                # Resolve WSL interface IP
                iface_ip = "N/A"
                try:
                    for iface_obj in self.discover_interfaces():
                        if iface_obj["name"] == f"wsl:{iface_name}":
                            iface_ip = iface_obj["ip"]
                            break
                except Exception:
                    pass

                # log_msg = (
                #     f"[WSL Capture] Interface: {iface_name} (IP: {iface_ip}) | "
                #     f"Packet: {pkt_data['src_ip']}:{pkt_data['src_port']} -> {pkt_data['dst_ip']}:{pkt_data['dst_port']} ({pkt_data['protocol']}) | "
                #     f"Total Packets: {self.packets_captured} | Total Flows: {self.flows_generated}"
                # )
                # logger.info(log_msg)
                # print(log_msg)

                completed = self._aggregator.add_packet(pkt_data)

                # Get the updated active flow state for real-time packet updates
                key = (
                    pkt_data["src_ip"],
                    pkt_data["dst_ip"],
                    pkt_data["src_port"],
                    pkt_data["dst_port"],
                    pkt_data["protocol"],
                )
                flow_state = None
                with self._aggregator._lock:
                    if key in self._aggregator._active_flows:
                        flow_state = dict(self._aggregator._active_flows[key])
                        if "tcp_flags" in flow_state:
                            flow_state["tcp_flags"] = set(flow_state["tcp_flags"])

                if flow_state:
                    event = {
                        "flow": flow_state,
                        "delta_bytes": pkt_data.get("length", 0),
                        "total_packets_captured": self.packets_captured,
                        "bytes_captured": self.bytes_captured
                    }
                    for listener in list(self.packet_listeners):
                        try:
                            listener(event)
                        except Exception as e:
                            logger.error(f"Error notifying packet listener: {e}")

                if completed:
                    self._emit_flow(completed)
            except Exception as e:
                logger.error(f"Error parsing WSL sniffer output line '{line}': {e}")

        # Check for errors on exit
        if proc.returncode is not None and proc.returncode != 0:
            stderr_content = proc.stderr.read()
            if stderr_content:
                logger.error(f"WSL sniffer exited with code {proc.returncode}. Stderr: {stderr_content}")
                self._errors.append(f"WSL exit: {stderr_content.strip()}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> dict:
        """Returns current capture status and counters."""
        return {
            "running": self._running,
            "interface": self.capture_interface,
            "packets_captured": self.packets_captured,
            "flows_generated": self.flows_generated,
            "bytes_captured": self.bytes_captured,
            "last_event_timestamp": self.last_event_timestamp,
            "capture_start_time": self.capture_start_time,
            "errors": list(self._errors[-5:]),  # Last 5 errors
            "scapy_available": _scapy_available,
            "scapy_error": _scapy_error,
        }

    @property
    def flow_queue(self) -> asyncio.Queue:
        """The asyncio queue from which the WebSocket pipeline reads flows."""
        return self._flow_queue

    @property
    def is_running(self) -> bool:
        return self._running

    def register_listener(self, callback):
        """Registers a callback to receive packet events in real-time."""
        if callback not in self.packet_listeners:
            self.packet_listeners.append(callback)

    def unregister_listener(self, callback):
        """Unregisters a callback from receiving packet events."""
        if callback in self.packet_listeners:
            self.packet_listeners.remove(callback)

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Sets the asyncio event loop to use for thread-safe queue operations."""
        self._loop = loop

    def _queue_put_on_loop(self, flow: dict):
        """Puts a flow onto the asyncio queue. Runs on the asyncio loop thread."""
        try:
            if self._flow_queue.full():
                try:
                    self._flow_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._flow_queue.put_nowait(flow)
        except Exception as e:
            logger.error(f"Error putting flow onto queue: {e}")

    # ------------------------------------------------------------------
    # Internal — packet callback and flow emission
    # ------------------------------------------------------------------
    def _on_packet(self, pkt):
        """
        Called by AsyncSniffer for every captured Windows packet.
        Runs in the sniffer thread.
        """
        try:
            parsed = parse_packet(pkt)
            if parsed is None:
                return

            self.packets_captured += 1
            self.bytes_captured += parsed["length"]
            self.last_event_timestamp = datetime.now(timezone.utc).isoformat()

            # Resolve Windows interface IP
            iface_ip = "N/A"
            try:
                iface_ip = get_if_addr(self.capture_interface)
            except Exception:
                pass

            # log_msg = (
            #     f"[Windows Capture] Interface: {self.capture_interface} (IP: {iface_ip}) | "
            #     f"Packet: {parsed['src_ip']}:{parsed['src_port']} -> {parsed['dst_ip']}:{parsed['dst_port']} ({parsed['protocol']}) | "
            #     f"Total Packets: {self.packets_captured} | Total Flows: {self.flows_generated}"
            # )
            # logger.info(log_msg)
            # print(log_msg)

            # Add to flow aggregator — may return a completed flow
            completed = self._aggregator.add_packet(parsed)

            # Notify per-packet listeners with raw packet info (fast path for live UI)
            if self.packet_listeners:
                event = {
                    "src_ip": parsed["src_ip"],
                    "dst_ip": parsed["dst_ip"],
                    "src_port": parsed["src_port"],
                    "dst_port": parsed["dst_port"],
                    "protocol": parsed["protocol"],
                    "delta_bytes": parsed["length"],
                    "total_packets_captured": self.packets_captured,
                    "bytes_captured": self.bytes_captured,
                }
                for listener in list(self.packet_listeners):
                    try:
                        listener(event)
                    except Exception as e:
                        logger.error(f"Error notifying packet listener: {e}")

            if completed:
                self._emit_flow(completed)

        except Exception as e:
            err_msg = f"Packet processing error: {e}"
            logger.error(err_msg)
            if len(self._errors) < 100:
                self._errors.append(err_msg)

    def _flush_loop(self):
        """Background thread that periodically flushes expired flows and logs capture statistics."""
        last_log_time = time.time()
        while not self._stop_event.is_set():
            try:
                expired_flows = self._aggregator.flush_expired()
                for flow in expired_flows:
                    self._emit_flow(flow)
            except Exception as e:
                logger.error(f"Flow flush error: {e}")
            
            # Periodically log capture statistics to stdout (every 4-5 seconds)
            now = time.time()
            if now - last_log_time >= 4.0:
                print(f"[PACKET_CAPTURE] packets={self.packets_captured} bytes={self.bytes_captured} flows={self.flows_generated}", flush=True)
                last_log_time = now

            self._stop_event.wait(timeout=2.0)

    def _emit_flow(self, flow: dict):
        """
        Puts a finalized flow dict onto the asyncio queue for consumption
        by the WebSocket alerts endpoint.
        """
        self.flows_generated += 1
        self.last_event_timestamp = datetime.now(timezone.utc).isoformat()
        self.recent_flows.append(flow)

        print(f"[FLOW_FEATURES] src={flow.get('src_ip')}:{flow.get('src_port')} dst={flow.get('dst_ip')}:{flow.get('dst_port')} proto={flow.get('protocol')} pkts={flow.get('packet_count')} bytes={flow.get('byte_count')} dur={flow.get('duration')} syn={flow.get('syn_count')} ack={flow.get('ack_count')}", flush=True)
        print(f"[FLOW_EMIT] flows_total={self.flows_generated} active_remaining={len(self._aggregator._active_flows)}", flush=True)

        try:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._queue_put_on_loop, flow)
            else:
                logger.error("Queue put skipped: event loop not set yet")
        except Exception as e:
            logger.error(f"Queue put error: {e}")


# ---------------------------------------------------------------------------
# Flow-to-Alert Converter — maps raw flow to the Cyclone alert schema
# ---------------------------------------------------------------------------
def flow_to_alert(
    flow: dict,
    rf_model=None,
    preprocessor=None,
    label_encoder=None,
    get_attack_metadata=None,
) -> dict:
    """
    Converts a raw aggregated flow dict into the CYCLONE alert JSON schema
    that the dashboard expects. Optionally runs ML classification.

    This function bridges the live collector's output format to the exact
    same schema used by test_lab_ingestion and mock generator.
    """
    import numpy as np
    import pandas as pd

    # Map observed flow to the ML model's 18 numerical + 3 categorical features
    # Features not directly observable from raw packets get sensible defaults
    protocol_str = (flow.get("protocol", "TCP") or "TCP").lower()
    if protocol_str == "tcp":
        proto_type = "tcp"
    elif protocol_str == "udp":
        proto_type = "udp"
    elif protocol_str == "icmp":
        proto_type = "icmp"
    else:
        proto_type = "tcp"

    dst_port = flow.get("dst_port", 0)
    # Map common ports to service names the model understands
    port_service_map = {
        80: "http", 443: "http", 8080: "http", 8443: "http",
        53: "domain_u", 25: "smtp", 587: "smtp",
        123: "ntp", 21: "ftp", 22: "auth", 23: "telnet",
    }
    service = port_service_map.get(dst_port, "other")

    # Determine TCP flag for the model
    syn_count = flow.get("syn_count", 0)
    ack_count = flow.get("ack_count", 0)
    rst_count = flow.get("rst_count", 0)
    fin_count = flow.get("fin_count", 0)

    if syn_count > 0 and ack_count > 0 and fin_count > 0:
        flag = "SF"  # Normal completed connection
    elif syn_count > 0 and rst_count > 0:
        flag = "RSTR"
    elif syn_count > 0 and ack_count == 0:
        flag = "S0"  # SYN sent, no response
    elif rst_count > 0:
        flag = "REJ"
    else:
        flag = "SF"

    packet_count = flow.get("packet_count", 1)
    byte_count = flow.get("byte_count", 0)
    duration = flow.get("duration", 0.001)

    ml_features = {
        "duration": float(duration),
        "src_bytes": int(byte_count * 0.4),  # Approximate split
        "dst_bytes": int(byte_count * 0.6),
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "num_compromised": 0,
        "count": packet_count,
        "srv_count": max(1, packet_count - 2),
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "dst_host_count": min(packet_count * 2, 255),
        "dst_host_srv_count": min(packet_count, 255),
        "dst_host_same_srv_rate": 0.95,
        "dst_host_diff_srv_rate": 0.02,
        "protocol_type": proto_type,
        "service": service,
        "flag": flag,
    }

    # Run ML classification if artifacts are loaded
    predicted_class = "Unknown"
    confidence = 0.0

    NUMERICAL_FEATURES = [
        "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "num_compromised", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "same_srv_rate", "diff_srv_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
    ]
    CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

    if rf_model is not None and preprocessor is not None and label_encoder is not None:
        try:
            df_flow = pd.DataFrame([ml_features])
            X_proc = preprocessor.transform(df_flow[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
            probabilities = rf_model.predict_proba(X_proc)[0]
            pred_idx = int(np.argmax(probabilities))
            # Verify feature compatibility, but do not assign attack labels to live traffic
            # raw_class = str(label_encoder.inverse_transform([pred_idx])[0]).replace("_", " ")
        except Exception as e:
            logger.warning(f"ML classification pipeline verification failed: {e}")

    # Build severity and explanation
    severity = "Low"
    explanation = [
        {
            "feature": "ML Classifier Mode",
            "impact": "0.0",
            "desc": "Classification is unavailable/unknown because the classifier is not trained on live traffic.",
        }
    ]

    return {
        "id": str(uuid.uuid4()),
        "timestamp": flow.get("capture_timestamp", datetime.now(timezone.utc).isoformat()),
        "src_ip": flow.get("src_ip", "0.0.0.0"),
        "dst_ip": flow.get("dst_ip", "0.0.0.0"),
        "src_port": flow.get("src_port", 0),
        "dst_port": dst_port,
        "protocol": flow.get("protocol", "TCP"),
        "duration": round(duration, 4),
        "packet_count": packet_count,
        "byte_count": byte_count,
        "avg_packet_size": flow.get("avg_packet_size", 0),
        "attack_type": predicted_class,
        "severity": severity,
        "confidence": round(confidence, 2),
        "data_source": "LIVE-NETWORK",
        "explanation": explanation,
        "syn_count": flow.get("syn_count", 0),
        "ack_count": flow.get("ack_count", 0),
        "rst_count": flow.get("rst_count", 0),
        "fin_count": flow.get("fin_count", 0),
    }
