# CyberForecaster — Live SOC Demonstration Runbook
**SIH 2026 / NTRO PS 26153 — AI-based Network Attack Forecasting from Network Traffic Data**

This runbook provides the verified, step-by-step procedure to demonstrate CyberForecaster to SIH evaluators. Follow these exact steps.

---

## 1. Prerequisites & Terminal Setup

Ensure you have 3 to 4 PowerShell or Windows Terminal tabs open:
- **Tab 1: Backend API & Socket Server**
- **Tab 2: Frontend SOC Dashboard**
- **Tab 3: Traffic Generator / Telemetry Ingestion**
- **Tab 4 (Optional): Local Hardhat Blockchain Ledger Node**

Verify environment dependencies:
```powershell
# In repo root:
py -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

---

## 2. Service Startup Sequence

### Step 1: [OPTIONAL] Local Blockchain Audit Ledger
If demonstrating cryptographic audit trail provenance:
```powershell
# Tab 4: Start local Hardhat Ethereum node
npx hardhat node
```
*In a separate sub-shell or before backend start:*
```powershell
# Deploy ForecastRegistry smart contract
npx hardhat run blockchain/scripts/deploy.js --network localhost
```
> **Note:** If Hardhat is not running, CyberForecaster automatically operates in **Ledger Offline** mode and truthfully displays provenance status without crashing or fabricating transactions.

---

### Step 2: [REQUIRED] Start CyberForecaster FastAPI Backend
```powershell
# Tab 1: Start backend from repo root
uvicorn backend.main:fastapi_app --host 127.0.0.1 --port 8000
```
**Expected Output:**
- `StateScaler loaded from models_weights/scaler.joblib`
- `Temporal LSTM World Model weights loaded from models_weights/lstm_world_model.pt`
- `ForecastExplainer engine initialized with active model and scaler`
- `Live Collector Socket.IO event pump task started`
- `Uvicorn running on http://127.0.0.1:8000`

Verify backend readiness in browser or curl:
```powershell
curl http://127.0.0.1:8000/api/world_model/status
```

---

### Step 3: [REQUIRED] Start Vite SOC Dashboard
```powershell
# Tab 2: Start frontend
cd frontend
npm run dev
```
Open **http://localhost:5173** in your browser.

**What to Observe on Initial Load:**
- **Status Indicators (Top Right):**
  - `FLOW DETECTOR: ACTIVE`
  - `SNORT: CONNECTED` (or `NOT CONNECTED` if Snort service is quiescent)
  - `LEDGER: ONLINE` (or `OFFLINE` if Hardhat is skipped)
  - `WORLD MODEL WARMING UP (0 / 10 Windows)`
  - `SOCKET: ONLINE`
- **Monitored Assets:** 5 enterprise endpoints displayed (`192.168.1.10`, etc.).
- **Empty States:** Clean, professional empty states stating *"Awaiting live telemetry"* and *"No active threat forecast detected"*. **Zero fabricated metrics.**

---

### Step 4: [OPTIONAL] Snort Live IDS Service
If Snort is installed locally on Windows (default path: `C:\Users\piyus\Snort\bin\snort.exe`):
```powershell
# Start Snort in fast alert mode monitoring active adapter
C:\Users\piyus\Snort\bin\snort.exe -c C:\Users\piyus\Snort\etc\snort.conf -l C:\Users\piyus\Snort\log -A fast -i 1
```
Verify Snort status:
```powershell
curl http://127.0.0.1:8000/api/snort/status
```

---

## 3. Live Demonstration Walkthrough

### Step 5: [REQUIRED] Ingest Network Telemetry
In the dashboard UI:
1. Navigate to **Packet Collector** (top right of Dashboard or Live Traffic tab).
2. Select your active network adapter from the dropdown.
3. Click **START CAPTURE**.
4. Generate benign local traffic (e.g., browse web pages, ping gateway, or run controlled test script):
   ```powershell
   # Tab 3: Controlled benign traffic generation
   py -c "import urllib.request, time; [urllib.request.urlopen('http://127.0.0.1:8000/api/hosts') and time.sleep(0.5) for _ in range(30)]"
   ```
5. Observe:
   - **Flow Activity Feed:** Real TCP/UDP/ICMP flows arriving via WebSocket in real time.
   - **Temporal Context Buffer:** Progress bar advancing window-by-window: `1/10 -> 2/10 -> ... -> 10/10 (READY)`.
   - **Multi-Step Horizon Trajectory:** Shows forecasted threat levels at `+5s`, `+10s`, `+15s`, `+20s`, `+25s`.

---

### Step 6: [REQUIRED] Demonstrate "WHY THIS FORECAST?" (Evidence Attribution)
Click on an active or anomalous host in the Monitored Assets list. Scroll to the **WHY THIS FORECAST?** panel:
1. **Tier 1: Model Evidence:**
   - Review spatio-temporal gradient saliency attributions across the 10 time windows.
   - Observe directional risk attribution (`↑ risk` / `↓ safe`).
2. **Tier 2: Network Telemetry:**
   - Review physical 23-D telemetry deviations vs the host's preceding baseline (strictly leak-free).
3. **Tier 3: Security Sensors:**
   - FastFlowDetector anomaly suspicion score and Snort signature alert (if matched).
4. **Lower Dimensions:**
   - **Stage Transition:** MITRE ATT&CK progression.
   - **MC-Dropout Predictive Uncertainty:** Empirical spread $[P_{\text{low}}, P_{\text{high}}]$ and $\sigma$ across 10 stochastic forward passes.
   - **Audit Provenance:** SHA-256 state hash and blockchain registration status.

---

### Step 7: [REQUIRED] Attack Path Reconstruction
Navigate to the **Topology** tab:
1. View the network topology map showing Monitored Assets connected to the Gateway.
2. Observe the **Attack Path Reconstruction** panel:
   - Displays reconstructed sequential hops (`A -> B -> C`).
   - Labeled clearly as: `OBSERVED TELEMETRY EVIDENCE (Multi-hop Reconstruction, not forecasted)`.
   - Each hop details source, destination, protocol, severity, evidence tag, and MITRE technique mapping.

---

### Step 8: [REQUIRED] Blockchain Audit Verification
Navigate to the **Audit Log** tab:
1. View the immutable list of non-benign threat forecasts submitted on-chain.
2. Click **Verify Authenticity** on any record:
   - Queries the smart contract `getForecast(forecastId)`.
   - Displays matching SHA-256 data hash, transaction receipt, and block number.
   - Explains: *"Proves record registration provenance in audit ledger; does not assert forecast ground-truth correctness."*

---

### Step 9: [REQUIRED] Validated Benchmark & Lead Time
Navigate to the **Forecast** tab and scroll to **Validated Benchmark Metrics**:
1. **Explain the Offline Held-Out Benchmark:**
   - Points to the canonical CSE-CIC-IDS2018 test split ($N=2,562$ sequences).
   - $K=1$ Future-State RMSE: **Temporal LSTM (2.05)** vs **Persistence (11.74)** vs **Train Mean (2.15)**.
   - Classification on untouched test partition: **Precision 96.92%**, **Recall 99.65%**, **$F_1$ Score 98.26%**, **FPR 6.29%**.
2. **Explain the Validated Lead Time:**
   - **0.0s (Exact-Onset Detection).**
   - **0 false alarms** on preceding benign baseline.
   - Explain honestly: *"The system detects attacks at the exact onset window; future attack-state prediction is not confused with pre-onset early warning."*

---

## 4. Troubleshooting & Graceful Degradation

| Issue | Normal SOC State | Recovery / Action |
|---|---|---|
| **Hardhat node not running** | Header displays `LEDGER: OFFLINE`. Predictions function normally. Verification modal states `NODE_OFFLINE`. | Optional. Start `npx hardhat node` if on-chain demonstration is needed. |
| **Snort not running** | Header displays `SNORT: NOT CONNECTED`. Explainability states `Snort: None correlated`. | System continues operating using flow telemetry + FastFlowDetector. |
| **No traffic flowing** | Shows `0 Flows Processed`, empty states render cleanly without invented numbers. | Start packet collector or run traffic script. |
| **Socket disconnected** | Header displays `SOCKET: DISCONNECTED`. Automatically attempts reconnect every 1s. | Verify backend is running on port 8000. |

---

## 5. Automated Demo Readiness Check
Run the pre-demo health check script before presenting to judges:
```powershell
py scripts/demo_health_check.py
```
If all checks return `[PASS]`, CyberForecaster is ready for live demonstration.
