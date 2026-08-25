# Technical Architecture Specification: CyberForecaster

## 1. Executive Summary

**CyberForecaster** is an AI-based Network Attack Forecasting System engineered for enterprise networks and Critical Information Infrastructure (CII). Unlike conventional Intrusion Detection Systems (IDS) operating as static binary classifiers ($ \text{Traffic} \to \text{Benign/Malicious} $), CyberForecaster learns the temporal state transition dynamics of observed network behavior:

$$ P(S[t+1] \mid S[t], S[t-1], \dots, S[t-L+1]) $$

By continuously projecting future network state vectors $S(t+1), \dots, S(t+K)$ forward in time, the system enables proactively forecasting cyberattacks before compromise completion.

---

## 2. Pipeline Overview & Data Flow

```text
 ┌─────────────────────────────────────────────────────────┐
 │ Input Telemetry (CSV / PCAP)                            │
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ Modular Ingestion Layer (csv_loader.py / pcap_parser.py)│
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ Feature Extractor & Time Windowing Engine (5s W-windows) │
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ 23-Dimensional Network State Vector Representation S(t)  │
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ Sequence Construction (L=10 historical windows)         │
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ PyTorch Temporal LSTM Multi-Task World Model            │
 └──────────────────────────┬──────────────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
 ┌─────▼──────────┐ ┌───────▼──────────┐ ┌───────▼──────────┐
 │ Head 1:        │ │ Head 2:          │ │ Head 3:          │
 │ Next State     │ │ Attack           │ │ ATT&CK Stage     │
 │ Vector S(t+1)  │ │ Probability P    │ │ Logits (0..5)    │
 └─────┬──────────┘ └──────────────────┘ └──────────────────┘
       │
 ┌─────▼───────────────────────────────────────────────────┐
 │ Recursive K-Step Forward Rollout Engine                 │
 │ S(t) -> S(t+1) -> S(t+2) -> ... -> S(t+K)               │
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ Explainability Engine (Gradient / Feature Attribution)  │
 └──────────────────────────┬──────────────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────────┐
 │ Streamlit SOC Dashboard                                 │
 └─────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Breakdown

### 3.1 Preprocessing & Encoding
- **CSV & PCAP Adapters:** Normalize flow/packet data into standardized internal schema.
- **State Encoder (`state_encoder.py`):** Aggregates windowed traffic into 23 features including packet/byte counts, unique IPs/ports, TCP/UDP/flag ratios, packet size stats, IAT stats, retransmission rates, TTL stats, inbound/outbound ratio, failed connection rate, and port entropy.
- **StateScaler (`scaler.py`):** Standardizes state dimensions using statistics fitted strictly on the training set to prevent target leakage.

### 3.2 PyTorch Temporal World Model (`models/lstm_world_model.py`)
- **Backbone:** 2-layer LSTM with hidden dimension 128 and dropout 0.2.
- **Heads:**
  - `state_head`: Predicts next state vector $S(t+1) \in \mathbb{R}^{23}$.
  - `attack_head`: Predicts scalar attack probability $P(\text{attack}_{t+1}) \in [0, 1]$.
  - `stage_head`: Predicts logits across 6 MITRE ATT&CK stages (Normal, Reconnaissance, Initial Access, Lateral Movement, C2, Exfiltration).

### 3.3 Forecasting & Risk Simulation (`forecasting/rollout.py`)
- Takes historical sequence $[S(t-9), \dots, S(t)]$, recursively predicts $\hat{S}(t+1)$, feeds $\hat{S}(t+1)$ back into model input, and simulates up to $K=5$ steps into the future.
