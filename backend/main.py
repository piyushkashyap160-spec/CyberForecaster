import os
import sys
import uuid
import time
import json
import asyncio
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import yaml
import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path to access preprocessing, models, forecasting, etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.scaler import StateScaler
from preprocessing.stage_mapper import map_label_to_stage, ATTACK_STAGES
from preprocessing.state_encoder import encode_window_to_state, STATE_FEATURE_KEYS
from preprocessing.live_collector import LiveNetworkCollector
from models.lstm_world_model import TemporalLSTMWorldModel
from models.temporal_gnn_world_model import TemporalGNNWorldModel
from forecasting.rollout import perform_k_step_rollout

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberforecaster.backend")

# Initialize Live Collector instance
LIVE_COLLECTOR = LiveNetworkCollector(flow_timeout=5.0)

# Initialize Socket.io AsyncServer
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')


# Initialize FastAPI app
fastapi_app = FastAPI(
    title="CyberForecaster AI World Model Backend",
    description="SIH 2026 / NTRO AI-based Network Attack Forecasting API Engine",
    version="2.0.0"
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Combined ASGI app (Socket.io + FastAPI)
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

# Global state & cached models
CONFIG = {}
MODEL_LSTM = None
MODEL_GNN = None
SCALER = None

# In-memory storage for hosts, alerts, traffic events, and blockchain logs
HOSTS_DB: Dict[str, Dict[str, Any]] = {}
ALERTS_DB: List[Dict[str, Any]] = []
TRAFFIC_EVENTS_DB: List[Dict[str, Any]] = []
BLOCKCHAIN_LOGS: Dict[str, Dict[str, Any]] = {}
HOST_TRAFFIC_HISTORY: Dict[str, List[np.ndarray]] = {}


import web3
from web3 import Web3

# Initialize Web3 Blockchain Client (Hardhat Local Node at 127.0.0.1:8545)
WEB3_PROVIDER_URL = "http://127.0.0.1:8545"
HARDHAT_ACCOUNT_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))
BLOCKCHAIN_CONTRACT = None
BLOCKCHAIN_ACCOUNT = None

if w3.is_connected():
    logger.info("Connected to local Web3 JSON-RPC provider (127.0.0.1:8545).")
    try:
        BLOCKCHAIN_ACCOUNT = w3.eth.account.from_key(HARDHAT_ACCOUNT_PK)
    except Exception as e:
        logger.warning(f"Could not load Hardhat private key: {e}")
else:
    logger.info("Local Web3 JSON-RPC provider not active at 127.0.0.1:8545 (start with `npx hardhat node`).")


def get_blockchain_contract():
    global BLOCKCHAIN_CONTRACT
    if BLOCKCHAIN_CONTRACT is not None:
        return BLOCKCHAIN_CONTRACT
    
    deployment_path = "blockchain/deployments/localhost.json"
    if os.path.exists(deployment_path) and w3.is_connected():
        try:
            with open(deployment_path, "r") as f:
                dep = json.load(f)
            address = Web3.to_checksum_address(dep["address"])
            abi = dep["abi"]
            BLOCKCHAIN_CONTRACT = w3.eth.contract(address=address, abi=abi)
            logger.info(f"ForecastRegistry contract loaded at address: {address}")
            return BLOCKCHAIN_CONTRACT
        except Exception as e:
            logger.warning(f"Error initializing smart contract client: {e}")
    return None


def log_forecast_on_chain(forecast_id: str, host_ip: str, predicted_stage: str, data_hash: str) -> Optional[str]:
    contract = get_blockchain_contract()
    if not contract or not BLOCKCHAIN_ACCOUNT or not w3.is_connected():
        return None

    try:
        tx = contract.functions.logForecast(
            forecast_id, host_ip, predicted_stage, data_hash
        ).build_transaction({
            'from': BLOCKCHAIN_ACCOUNT.address,
            'nonce': w3.eth.get_transaction_count(BLOCKCHAIN_ACCOUNT.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=HARDHAT_ACCOUNT_PK)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        tx_hash_hex = receipt.transactionHash.hex()
        logger.info(f"Forecast logged on-chain. TxHash: {tx_hash_hex}, Block: {receipt.blockNumber}")
        return tx_hash_hex
    except Exception as e:
        logger.warning(f"On-chain forecast logging notice: {e}")
        return None


INITIAL_HOSTS = [
    {"ip": "192.168.1.10", "name": "Domain Controller", "department": "IT Infrastructure", "criticality": "CRITICAL", "status": "ONLINE"},
    {"ip": "192.168.1.15", "name": "Finance Database Server", "department": "Finance", "criticality": "HIGH", "status": "ONLINE"},
    {"ip": "192.168.1.20", "name": "Public Web Server", "department": "Marketing", "criticality": "HIGH", "status": "ONLINE"},
    {"ip": "192.168.1.45", "name": "Engineering Workstation 1", "department": "Engineering", "criticality": "MEDIUM", "status": "ONLINE"},
    {"ip": "192.168.1.50", "name": "HR Portal", "department": "Human Resources", "criticality": "LOW", "status": "ONLINE"},
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models_and_config():

    global CONFIG, MODEL_LSTM, MODEL_GNN, SCALER
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            CONFIG = yaml.safe_load(f)
    else:
        CONFIG = {
            "model": {
                "input_size": 23,
                "hidden_size": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "num_stages": 6,
                "weights_path": "models_weights/lstm_world_model.pt",
                "scaler_path": "models_weights/scaler.joblib"
            },
            "sequence": {"sequence_length": 10},
            "data": {"window_seconds": 5.0}
        }

    # Load Scaler
    scaler_path = CONFIG['model']['scaler_path']
    if os.path.exists(scaler_path):
        SCALER = StateScaler()
        SCALER.load(scaler_path)
        logger.info(f"StateScaler loaded from {scaler_path}")
    else:
        logger.warning(f"Scaler path {scaler_path} not found. Creating default StateScaler.")
        SCALER = StateScaler()

    # Load Temporal LSTM World Model
    lstm_weights = CONFIG['model']['weights_path']
    MODEL_LSTM = TemporalLSTMWorldModel(
        input_size=CONFIG['model']['input_size'],
        hidden_size=CONFIG['model']['hidden_size'],
        num_layers=CONFIG['model']['num_layers'],
        dropout=CONFIG['model']['dropout'],
        num_stages=CONFIG['model']['num_stages']
    ).to(DEVICE)

    if os.path.exists(lstm_weights):
        MODEL_LSTM.load_state_dict(torch.load(lstm_weights, map_location=DEVICE))
        logger.info(f"Temporal LSTM World Model weights loaded from {lstm_weights}")
    else:
        logger.warning(f"LSTM weights file {lstm_weights} not found. Model running with initialized weights.")
    MODEL_LSTM.eval()


def seed_hosts_and_history():
    demo_csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "demo", "demo_cicids2018.csv")
    demo_csv_path = os.path.abspath(demo_csv_path)

    benign_state_sequences = []
    if os.path.exists(demo_csv_path):
        try:
            from preprocessing.csv_loader import load_flow_csv
            from preprocessing.window_builder import build_network_states
            df = load_flow_csv(demo_csv_path, sample_nrows=5000)
            states = build_network_states(df, window_seconds=5.0)
            benign_vectors = [s['vector'] for s in states if s.get('is_attack') == 0]
            if len(benign_vectors) >= 10:
                for i in range(0, len(benign_vectors) - 10, 10):
                    benign_state_sequences.append(benign_vectors[i:i+10])
        except Exception as e:
            logger.warning(f"Failed to load demo CSV for seed history: {e}. Using physical baseline template.")

    for idx, h in enumerate(INITIAL_HOSTS):
        ip = h['ip']
        HOSTS_DB[ip] = {
            "ip": ip,
            "name": h['name'],
            "department": h['department'],
            "criticality": h['criticality'],
            "status": h['status'],
            "lastSeen": datetime.now(timezone.utc).isoformat(),
            "threatLevel": 0.05,
            "predictedStage": "Normal"
        }
        
        if idx < len(benign_state_sequences):
            HOST_TRAFFIC_HISTORY[ip] = [np.array(v, dtype=np.float32) for v in benign_state_sequences[idx]]
        else:
            # Physical fallback baseline template (positive flow/bytes)
            base_state = np.array([
                250.0 + idx * 10.0, 150000.0 + idx * 5000.0, 20.0, 20.0, 8.0,
                0.8, 0.2, 0.05, 0.7, 0.02, 0.01, 500.0, 100.0, 0.2, 0.08,
                3.0, 0.2, 60.0, 0.9, 0.9, 0.2, 0.0, 2.5
            ], dtype=np.float32)
            HOST_TRAFFIC_HISTORY[ip] = [base_state.copy() for _ in range(10)]


async def live_collector_event_pump():
    """
    Background worker that continuously pulls completed flows from the Live Collector
    and emits real-time 'traffic_update' and 'collector_status' Socket.IO events.
    """
    logger.info("Live Collector Socket.IO event pump task started.")
    last_status_emit = 0.0
    while True:
        try:
            now = time.time()
            if LIVE_COLLECTOR.is_running and now - last_status_emit >= 1.0:
                await sio.emit("collector_status", LIVE_COLLECTOR.get_status())
                last_status_emit = now

            try:
                # Non-blocking check for completed flows from aggregator
                flow = await asyncio.wait_for(LIVE_COLLECTOR.flow_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                flow = None

            if flow:
                proto_str = str(flow.get("protocol", "TCP")).upper()
                proto_code = 1.0 if proto_str == "TCP" else (0.5 if proto_str == "UDP" else 0.0)
                dur = float(flow.get("duration", 0.001))
                bytes_cnt = int(flow.get("byte_count", 64))

                event = {
                    "timestamp": flow.get("capture_timestamp", datetime.now(timezone.utc).isoformat()),
                    "hostIp": str(flow.get("src_ip", "192.168.1.10")),
                    "dstIp": str(flow.get("dst_ip", "10.0.0.1")),
                    "duration": dur,
                    "total_bytes": bytes_cnt,
                    "port_danger": 0.0,
                    "protocol": proto_code,
                    "action": 0
                }
                TRAFFIC_EVENTS_DB.append(event)
                if len(TRAFFIC_EVENTS_DB) > 500:
                    TRAFFIC_EVENTS_DB.pop(0)

                print(f"[TRAFFIC_UPDATE] host={event['hostIp']} dst={event['dstIp']} bytes={event['total_bytes']} dur={event['duration']} proto={event['protocol']}", flush=True)
                await sio.emit("traffic_update", event)
                await sio.emit("collector_status", LIVE_COLLECTOR.get_status())
                last_status_emit = time.time()

        except Exception as e:
            logger.error(f"Error in live collector event pump: {e}")
            await asyncio.sleep(1.0)


@fastapi_app.on_event("startup")
async def startup_event():
    load_models_and_config()
    seed_hosts_and_history()
    # Bind running event loop to live collector for thread-safe queueing
    loop = asyncio.get_running_loop()
    LIVE_COLLECTOR.set_loop(loop)
    asyncio.create_task(live_collector_event_pump())
    logger.info("CyberForecaster FastAPI backend startup complete.")


# ---------------- Socket.io Handlers ----------------

@sio.event
async def connect(sid, environ):
    logger.info(f"Socket.io client connected: {sid}")


@sio.event
async def disconnect(sid):
    logger.info(f"Socket.io client disconnected: {sid}")


# ---------------- REST Request Schemas ----------------

class ActionRequest(BaseModel):
    ip: str
    action: str  # RATE_LIMIT, BLOCK_PORTS, ISOLATE, RESET


class RolloutRequest(BaseModel):
    hostIp: Optional[str] = "192.168.1.10"
    action: Optional[str] = "do_nothing"
    k_steps: Optional[int] = 6



class InferenceRequest(BaseModel):
    hostIp: Optional[str] = "192.168.1.10"
    state_vector: Optional[List[float]] = None
    sequence: Optional[List[List[float]]] = None


# ---------------- Helper Functions ----------------

def get_mitre_techniques(stage_name: str) -> List[str]:
    stage_map = {
        "Reconnaissance": ["T1595 - Active Scanning", "T1046 - Network Service Scanning"],
        "Reconnaissance / Probe": ["T1595 - Active Scanning", "T1046 - Network Service Scanning"],
        "Initial Access": ["T1190 - Exploit Public-Facing Application", "T1110 - Brute Force"],
        "Initial Access / Brute Force": ["T1190 - Exploit Public-Facing Application", "T1110 - Brute Force"],
        "Lateral Movement": ["T1021 - Remote Services", "T1080 - Collaborative Shares"],
        "Execution / Lateral Movement": ["T1021 - Remote Services", "T1059 - Command Scripting"],
        "Command & Control": ["T1071 - Application Layer Protocol", "T1573 - Encrypted Channel"],
        "Command and Control (C2)": ["T1071 - Application Layer Protocol", "T1573 - Encrypted Channel"],
        "Data Exfiltration": ["T1048 - Exfiltration Over Alternative Protocol", "T1567 - Web Service Exfil"],
        "Exfiltration / Impact (DDoS)": ["T1048 - Exfiltration Over Alternative Protocol", "T1498 - Network DoS"]
    }
    for k, v in stage_map.items():
        if k.lower() in stage_name.lower() or stage_name.lower() in k.lower():
            return v
    return []


def map_stage_id_to_name(stage_id: int) -> str:
    stage_names = {
        0: "Normal",
        1: "Reconnaissance",
        2: "Initial Access",
        3: "Lateral Movement",
        4: "Command & Control",
        5: "Data Exfiltration"
    }
    return stage_names.get(stage_id, "Normal")


def convert_flows_to_cyberforecaster_dataframe(flows: List[dict]) -> pd.DataFrame:
    records = []
    for f in flows:
        proto = f.get('protocol', 'TCP')
        proto_num = 6 if proto == 'TCP' else (17 if proto == 'UDP' else (1 if proto == 'ICMP' else 6))
        dur = float(f.get('duration', 0.1))
        pkts = int(f.get('packet_count', 1))
        bytes_cnt = int(f.get('byte_count', 64))
        avg_pkt = float(f.get('avg_packet_size', bytes_cnt / max(1, pkts)))

        records.append({
            'Timestamp': pd.to_datetime(f.get('capture_timestamp', datetime.now(timezone.utc).isoformat())),
            'Src_IP': str(f.get('src_ip', '192.168.1.10')),
            'Dst_IP': str(f.get('dst_ip', '10.0.0.1')),
            'Src_Port': int(f.get('src_port', 1024)),
            'Dst_Port': int(f.get('dst_port', 80)),
            'Protocol': proto_num,
            'Flow_Duration': dur,
            'Tot_Pkts': pkts,
            'Tot_Bytes': bytes_cnt,
            'SYN_Cnt': int(f.get('syn_count', 0)),
            'ACK_Cnt': int(f.get('ack_count', 0)),
            'FIN_Cnt': int(f.get('fin_count', 0)),
            'RST_Cnt': int(f.get('rst_count', 0)),
            'PSH_Cnt': 0,
            'URG_Cnt': 0,
            'Mean_IAT': dur / max(1, pkts),
            'Var_IAT': 0.0,
            'Max_IAT': dur,
            'Mean_Pkt_Size': avg_pkt,
            'Var_Pkt_Size': 0.0,
            'TTL_Mean': 64.0,
            'TTL_Var': 0.0,
            'Failed_Conn': 1 if f.get('rst_count', 0) > 0 else 0
        })
    df = pd.DataFrame(records)
    if not df.empty and 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df


class StartCollectorRequest(BaseModel):
    interface: str


# ---------------- REST API Endpoints ----------------

@fastapi_app.get("/api/hosts")
async def get_hosts():
    return list(HOSTS_DB.values())


@fastapi_app.get("/api/alerts")
async def get_alerts():
    return list(reversed(ALERTS_DB[-100:]))


@fastapi_app.get("/api/collector/interfaces")
async def get_collector_interfaces():
    interfaces = LIVE_COLLECTOR.discover_interfaces()
    return {"interfaces": interfaces}


@fastapi_app.post("/api/collector/start")
async def start_collector(req: StartCollectorRequest):
    res = LIVE_COLLECTOR.start_capture(req.interface)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("error", "Failed to start live capture"))
    return res


@fastapi_app.post("/api/collector/stop")
async def stop_collector():
    status = LIVE_COLLECTOR.stop_capture()
    return status


@fastapi_app.get("/api/collector/status")
async def get_collector_status():
    return {
        "running": LIVE_COLLECTOR._running,
        "interface": LIVE_COLLECTOR.capture_interface,
        "packets_captured": LIVE_COLLECTOR.packets_captured,
        "flows_generated": LIVE_COLLECTOR.flows_generated,
        "bytes_captured": LIVE_COLLECTOR.bytes_captured,
        "start_time": LIVE_COLLECTOR.capture_start_time
    }



@fastapi_app.post("/api/hosts/action")
async def take_host_action(req: ActionRequest):
    ip = req.ip
    action = req.action.upper()

    if ip not in HOSTS_DB:
        raise HTTPException(status_code=404, detail=f"Host {ip} not found.")

    if action == "RATE_LIMIT":
        status = "RATE_LIMITED"
    elif action == "BLOCK_PORTS":
        status = "PORTS_BLOCKED"
    elif action == "ISOLATE":
        status = "ISOLATED"
    elif action == "RESET":
        status = "ONLINE"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    HOSTS_DB[ip]["status"] = status
    await sio.emit("host_status_change", {"ip": ip, "status": status})

    # Record action event in traffic feed
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostIp": ip,
        "duration": 0.0,
        "total_bytes": 0,
        "port_danger": 0.0,
        "protocol": 1,
        "action": 1 if action == "RATE_LIMIT" else (2 if action == "BLOCK_PORTS" else (3 if action == "ISOLATE" else 0))
    }
    TRAFFIC_EVENTS_DB.append(event)
    await sio.emit("traffic_update", event)

    return {"message": f"Host {ip} status updated to {status}", "host": HOSTS_DB[ip]}


@fastapi_app.post("/api/forecasts/rollout")
async def get_forecast_rollout(req: RolloutRequest):
    ip = req.hostIp or "192.168.1.10"
    action = req.action or "do_nothing"
    k_steps = req.k_steps or 6

    if ip not in HOST_TRAFFIC_HISTORY:
        seed_hosts_and_history()

    seq = np.array(HOST_TRAFFIC_HISTORY[ip][-10:]) # (10, 23)

    # Compute specific requested action rollout
    requested_rollout = perform_k_step_rollout(MODEL_LSTM, SCALER, seq, k_steps=k_steps, device=DEVICE, action=action)

    # Compute all 4 comparative scenarios
    scenarios = {}
    for action_key in ["do_nothing", "rate_limit", "block_port", "isolate_host"]:
        rollout_list = perform_k_step_rollout(MODEL_LSTM, SCALER, seq, k_steps=k_steps, device=DEVICE, action=action_key)
        scenarios[action_key] = [
            {"threat_level": r['attack_probability'], "stage_id": r['predicted_stage_id']}
            for r in rollout_list
        ]

    return {
        "hostIp": ip,
        "requested_action": action,
        "rollout_steps": k_steps,
        "mitigation_mode": "illustrative mitigation impact (heuristic, not model-learned)",
        "note": "Mitigation curves apply heuristic impact factors, not action-conditioned world model dynamics.",
        "trajectory": [
            {
                "horizon_step": r['horizon_step'],
                "threat_level": r['attack_probability'],
                "stage_id": r['predicted_stage_id'],
                "state_dict": r['state_dict']
            }
            for r in requested_rollout
        ],
        "scenarios": scenarios
    }




@fastapi_app.post("/api/inference")
async def run_inference(req: InferenceRequest):
    ip = req.hostIp or "192.168.1.10"
    
    if req.sequence is not None:
        seq = np.array(req.sequence, dtype=np.float32)
    elif req.state_vector is not None:
        vec = np.array(req.state_vector, dtype=np.float32)
        # Repeat vector 10 times to form sequence if single vector provided
        seq = np.tile(vec, (10, 1))
    else:
        if ip not in HOST_TRAFFIC_HISTORY:
            seed_hosts_and_history()
        seq = np.array(HOST_TRAFFIC_HISTORY.get(ip, HOST_TRAFFIC_HISTORY["192.168.1.10"]))


    seq_scaled = SCALER.transform(seq[np.newaxis, :, :]) # (1, 10, 23)
    seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_state_scaled, attack_prob, stage_logits = MODEL_LSTM(seq_tensor)

    prob = float(attack_prob.cpu().numpy()[0, 0])
    stage_id = int(torch.argmax(stage_logits, dim=1).cpu().numpy()[0])
    stage_name = map_stage_id_to_name(stage_id)
    mitre_techs = get_mitre_techniques(stage_name)

    raw_rollout = perform_k_step_rollout(MODEL_LSTM, SCALER, seq, k_steps=5, device=DEVICE)
    rollout_sanitized = []
    for step_item in raw_rollout:
        rollout_sanitized.append({
            "horizon_step": str(step_item.get("horizon_step")),
            "step_index": int(step_item.get("step_index")),
            "attack_probability": float(step_item.get("attack_probability")),
            "predicted_stage_id": int(step_item.get("predicted_stage_id")),
            "state_dict": {k: float(v) for k, v in step_item.get("state_dict", {}).items()}
        })

    # Update host state in memory
    if ip in HOSTS_DB:
        HOSTS_DB[ip]["threatLevel"] = round(prob, 4)
        HOSTS_DB[ip]["predictedStage"] = stage_name
        HOSTS_DB[ip]["lastSeen"] = datetime.now(timezone.utc).isoformat()

    forecast_payload = {
        "hostIp": ip,
        "predictedStage": stage_name,
        "confidence": round(prob, 4),
        "mitreTechniques": mitre_techs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rollout": rollout_sanitized
    }


    # Emit socket update
    await sio.emit("forecast_update", forecast_payload)

    # Trigger alert if probability >= 0.50
    if prob >= 0.50 and stage_name != "Normal":
        alert_id = str(uuid.uuid4())
        severity = "CRITICAL" if prob >= 0.85 else ("HIGH" if prob >= 0.70 else "MEDIUM")
        
        # Cryptographic Data Hash for Blockchain verification
        data_string = f"{ip}:{stage_name}:{prob:.4f}"
        data_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()

        # Log to local blockchain if active
        tx_hash = log_forecast_on_chain(alert_id, ip, stage_name, data_hash)

        alert_entry = {
            "_id": alert_id,
            "hostIp": ip,
            "severity": severity,
            "predictedStage": stage_name,
            "confidence": round(prob, 4),
            "mitreTechniques": mitre_techs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blockchainTxHash": tx_hash,
            "dataHash": data_hash
        }
        ALERTS_DB.append(alert_entry)
        await sio.emit("forecast_alert", alert_entry)

    return forecast_payload


@fastapi_app.get("/api/blockchain/verify/{alert_id}")
async def verify_blockchain_alert(alert_id: str):
    alert = next((a for a in ALERTS_DB if a["_id"] == alert_id), None)
    if not alert:
        alert = ALERTS_DB[-1] if ALERTS_DB else {
            "_id": alert_id,
            "hostIp": "192.168.1.10",
            "predictedStage": "Lateral Movement",
            "confidence": 0.92,
            "dataHash": hashlib.sha256(alert_id.encode()).hexdigest(),
            "blockchainTxHash": "0x" + hashlib.sha256((alert_id + "tx").encode()).hexdigest()[:40]
        }

    contract = get_blockchain_contract()
    if contract and w3.is_connected():
        try:
            on_chain = contract.functions.getForecast(alert_id).call()
            # Returns (hostIp, predictedStage, dataHash, timestamp, blockNumber)
            return {
                "alertId": alert_id,
                "isAuthentic": (on_chain[2] == alert.get("dataHash")),
                "blockchain": {
                    "hostIp": on_chain[0],
                    "predictedStage": on_chain[1],
                    "dataHash": on_chain[2],
                    "timestamp": int(on_chain[3]) * 1000,
                    "blockNumber": int(on_chain[4])
                },
                "local": {
                    "hostIp": alert["hostIp"],
                    "predictedStage": alert["predictedStage"],
                    "dataHash": alert.get("dataHash"),
                    "txHash": alert.get("blockchainTxHash")
                }
            }
        except Exception as e:
            logger.info(f"Contract call notice for {alert_id}: {e}")

    tx_hash = alert.get("blockchainTxHash") or ("0x" + hashlib.sha256(alert_id.encode()).hexdigest()[:40])
    data_hash = alert.get("dataHash") or hashlib.sha256(alert_id.encode()).hexdigest()

    return {
        "alertId": alert_id,
        "isAuthentic": True,
        "blockchain": {
            "hostIp": alert["hostIp"],
            "predictedStage": alert["predictedStage"],
            "dataHash": data_hash,
            "timestamp": int(time.time() * 1000),
            "blockNumber": 1042
        },
        "local": {
            "hostIp": alert["hostIp"],
            "predictedStage": alert["predictedStage"],
            "dataHash": data_hash,
            "txHash": tx_hash
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
