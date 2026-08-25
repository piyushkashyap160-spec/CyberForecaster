# CyberForecaster: AI-Based Network Attack Forecasting System

**Tagline:** From Intrusion Detection to Attack Forecasting  
**SIH 2026 Problem Statement:** AI based Network Attack Forecasting from Network Traffic Data  
**Target Organization:** National Technical Research Organisation (NTRO)

---

## 1. System Architecture & Core Paradigm

Unlike traditional Intrusion Detection Systems (IDS) that act as static binary classifiers ($ \text{Traffic} \to \text{Benign/Malicious} $), **CyberForecaster** models the temporal dynamics of network state transitions:

$$ P(S[t+1] \mid S[t], S[t-1], \dots, S[t-L+1]) $$

The temporal world model learns how observed network states evolve over time. The core prediction pipeline operates as:

```text
  Raw Traffic (CSV / PCAP)
           ↓
   Feature Extractor
           ↓
  5s Time-Windowing & State Aggregation
           ↓
  Network State Sequence S(t-9)...S(t)
           ↓
   LSTM Temporal World Model
           ↓
┌─────────────────────────────────────────┐
│ Multi-Task Heads:                       │
│ 1. Next State Prediction S(t+1)         │
│ 2. Future Attack Probability P(Attack)  │
│ 3. MITRE ATT&CK Stage Classification    │
└─────────────────────────────────────────┘
           ↓
 Recursive K-Step Forward Rollout [S(t+1) ... S(t+K)]
           ↓
 Model Explainability (SHAP / Feature Attribution)
           ↓
 Interactive SOC Dashboard (Streamlit)
```

---

## 2. Environment & System Inspection Results

- **Operating System:** Windows (AMD64)
- **Python Version:** 3.14.x
- **GPU / Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, CUDA Driver 13.3)
- **Free Disk Space:** ~246 GB on `C:`
- **Core Dependencies:** PyTorch, Scikit-learn, Pandas, NumPy, Scapy, Streamlit, Plotly, PyYAML, SHAP, Pytest.

---

## 3. Detailed Component Specification

### 3.1 Data Ingestion & Preprocessing
- **CSV Adapter:** Supports CIC-IDS2018, CIC-IDS2017, UNSW-NB15 flow CSV formats. Standardizes flow fields into common schema.
- **PCAP Adapter:** Parsed using Scapy / dpkt to extract flow & packet-level features (TTL variance, TCP window, TCP flags, packet IAT).
- **Time Windowing Engine:** Groups traffic into non-overlapping or sliding 5-second windows ($W=5\text{s}$).
- **State Encoder & Scaler:** Aggregates windowed traffic into a 23-dimensional normalized network state vector $S(t)$. Scaler is fitted strictly on training split to prevent target leakage.

### 3.2 Feature Schema $S(t)$
1. `total_packets`
2. `total_bytes`
3. `unique_src_ips`
4. `unique_dst_ips`
5. `unique_dst_ports`
6. `tcp_ratio`
7. `udp_ratio`
8. `syn_ratio`
9. `ack_ratio`
10. `rst_ratio`
11. `fin_ratio`
12. `mean_packet_size`
13. `packet_size_variance`
14. `mean_IAT`
15. `IAT_variance`
16. `max_IAT`
17. `retransmission_rate`
18. `ttl_mean`
19. `ttl_variance`
20. `inbound_outbound_ratio`
21. `failed_connection_rate`
22. `port_entropy`
23. `connection_rate`

### 3.3 Model Design & Multi-Task Heads
- **Primary World Model:** Multi-layer LSTM (`hidden_size=128`, `num_layers=2`, `dropout=0.2`).
- **Input:** Sequence of $L=10$ historical network states $[S(t-9), \dots, S(t)]$.
- **Outputs:**
  - **Head 1 (State Predictor):** Predicts $S(t+1) \in \mathbb{R}^{23}$ (MSE Loss).
  - **Head 2 (Attack Probability):** Predicts $P(\text{attack} \mid S_{t+1}) \in [0, 1]$ (BCE Loss).
  - **Head 3 (Attack Stage):** Predicts MITRE ATT&CK stage (CrossEntropy Loss):
    - `0: Normal`
    - `1: Reconnaissance`
    - `2: Initial Access`
    - `3: Lateral Movement`
    - `4: Command & Control`
    - `5: Exfiltration`
- **Recursive K-Step Rollout:** Feed predicted $\hat{S}(t+1)$ back into model to project $\hat{S}(t+2), \dots, \hat{S}(t+K)$ for $K=5$ horizon steps.

### 3.4 Baselines & Evaluation Metrics
- **Baseline:** Logistic Regression model trained on aggregated window features.
- **Metrics:** Precision, Recall, F1-Score, False Positive Rate (FPR), Mean Absolute Error (MAE) for state prediction, and Forecast Lead Time (time delta between early forecast threshold breach and actual attack onset).
- **Split Strategy:** Chronological split (70% train, 15% val, 15% test) to prevent temporal data leakage.

### 3.5 Model Explainability
- Feature attribution & SHAP integration to provide security analysts with exact top-contributing features (e.g., port entropy, SYN ratio spikes, IAT variance drop) for high-risk forecasted states.

### 3.6 Streamlit SOC Dashboard
- **Page 1: Overview:** Real-time risk status, forecast horizon, active warnings.
- **Page 2: Attack Forecast:** Multi-step forward probability timeline & state trajectories.
- **Page 3: Attack Progression:** Interactive MITRE ATT&CK stage pipeline highlighting active/predicted stage.
- **Page 4: Explainability:** SHAP feature importance & key security indicators.
- **Page 5: Traffic Explorer:** Detailed packet/flow level metrics table & statistics.
- **Page 6: Model Performance:** Benchmarks (LSTM World Model vs. Logistic Regression Baseline).

---

## 4. Implementation Milestones (P0 to P1)

1. **Phase 1: Project Setup & Synthetic/Demo Data Generation** (Realistic network attack flow dataset mimicking CIC-IDS2018 scenario).
2. **Phase 2: Data Pipeline & State Encoders** (`csv_loader.py`, `pcap_parser.py`, `window_builder.py`, `scaler.py`).
3. **Phase 3: Deep World Model & Baselines** (`lstm_world_model.py`, `train_world_model.py`, `train_baseline.py`).
4. **Phase 4: Multi-Step Rollout & Risk Engine** (`rollout.py`, `risk_engine.py`, `stage_mapping.py`, `lead_time.py`).
5. **Phase 5: Explainability Engine** (`shap_explainer.py`, `feature_attribution.py`).
6. **Phase 6: SOC Streamlit Dashboard** (`dashboard/app.py`).
7. **Phase 7: Comprehensive Testing & Verification** (`pytest` test suite, full end-to-end smoke test).
8. **Phase 8: Technical Documentation** (`README.md`, `docs/architecture.md`, `docs/methodology.md`, `docs/dataset.md`).

---

## 5. Assumptions & Limitations
- **Scientific Honesty:** The model learns temporal transition dynamics of observed network states $P(S[t+1] \mid S[t])$. It does not claim to discover immutable causal mechanics of cyberattacks.
- **Offline Operating Mode:** Inference operates entirely offline without external cloud dependencies.
- **Extensibility:** Architectural split between PCAP/CSV ingestion and Feature Schema allows easy addition of new dataset adapters (e.g., CTU-13, UNSW-NB15).
