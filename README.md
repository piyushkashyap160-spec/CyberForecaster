# CyberForecaster: AI-Based Network Attack Forecasting System

> **Tagline:** *From Intrusion Detection to Attack Forecasting*  
> **SIH 2026 Problem Statement:** AI based Network Attack Forecasting from Network Traffic Data  
> **Target Organization:** National Technical Research Organisation (NTRO)

---

## 1. Project Overview

**CyberForecaster** is an open-source, offline AI prototype engineered to model temporal network state dynamics and forecast cyberattacks before initial compromise is completed. Designed for Critical Information Infrastructure (CII) and enterprise Security Operations Centers (SOC), CyberForecaster shifts cybersecurity defense from reactive intrusion detection to proactive attack forecasting.

---

## 2. Problem Statement

Modern nation-state cyberattacks and Advanced Persistent Threats (APTs) operate via multi-stage kill chains. Traditional intrusion detection systems (IDS) trigger alerts only after malicious payloads or unauthorized access events have already occurred. NTRO's SIH 2026 problem statement requires an open-source AI system that learns network behavior, models temporal state dynamics, predicts future network states, and maps forecasted activity to ATT&CK-aligned stages with interpretable explanations.

---

## 3. Proposed Solution: The World Model Approach

CyberForecaster replaces static classification with a **Temporal World Model** learning state dynamics:

$$ P(S[t+1] \mid S[t], S[t-1], \dots, S[t-L+1]) $$

```text
CyberForecaster Paradigm:
Traffic History [S(t-9) ... S(t)]
       ↓
Current Network State S(t)
       ↓
Learn Temporal Transition Dynamics
       ↓
Predict Next State S(t+1)
       ↓
Recursive K-Step Forward Rollout [S(t+1) ... S(t+K)]
       ↓
Forecast Future Attack Risk & ATT&CK-Aligned Stage
       ↓
Explainable Early Warning Threshold Alert (Gradient Saliency)
```

---

## 4. Benchmark Evaluation & Performance (4 Models)

### 4-Model Comparative Benchmark Matrix (Hardened Synthetic Demo Dataset — 14,400 Flows)

| Metric / Model | Baseline A (Static LR S(t)) | Baseline B (Temporal LR S(t-9)...S(t)) | Proposed (Temporal LSTM World Model) | Experimental (Temporal GNN World Model) |
|---|---|---|---|---|
| **Precision** | 0.9444 | 0.9423 | **0.9444** | 0.9444 |
| **Recall** | 0.9444 | 0.9074 | **0.9444** | 0.9444 |
| **F1-Score** | 0.9444 | 0.9245 | **0.9444** | 0.9444 |
| **False Positive Rate (FPR)** | 0.0556 (5.56%) | 0.0556 (5.56%) | **0.0556 (5.56%)** | 0.0556 (5.56%) |
| **Confusion Matrix (TP/FP/TN/FN)** | 51 / 3 / 51 / 3 | 49 / 3 / 51 / 5 | **51 / 3 / 51 / 3** | 51 / 3 / 51 / 3 |
| **Next-State MAE** | N/A | N/A | **0.3864** | 0.3355 |
| **Next-State RMSE** | N/A | N/A | **0.6565** | 0.6343 |
| **Forecast Lead Time** | -5.0s | -5.0s | **-5.0s** | -5.0s |

---

## 5. Real CIC-IDS2018 Benchmark (1,048,575 Real Flows)

The pipeline [`training/run_cicids2018_pipeline.py`](file:///C:/Users/piyus/.gemini/antigravity/scratch/CyberForecaster/training/run_cicids2018_pipeline.py) evaluated all 4 models on the official `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` dataset (**1,048,575 flows**, 8,640 five-second windows, 8,630 sequences with 70/15/15 chronological split):

| Model | Precision | Recall | F1-Score | FPR | Next-State MAE | Next-State RMSE |
|---|---|---|---|---|---|---|
| **Static LR (Baseline A)** | 1.0000 | 0.9558 | 0.9774 | 0.00% | N/A | N/A |
| **Temporal LR (Baseline B)** | 0.9969 | 0.9868 | 0.9918 | 80.0% | N/A | N/A |
| **Temporal LSTM World Model** | 1.0000 | 0.2264 | 0.3692 | 0.00% | 0.6044 | 0.8730 |
| **Temporal GNN World Model** | 1.0000 | 0.4574 | 0.6277 | 0.00% | 0.6002 | 0.8865 |

---

## 6. Horizon-Wise K-Step Forecasting Performance

- **Sequence Length:** $L = 10$ historical windows ($50$ seconds context).
- **Forecast Horizon:** $K = 5$ steps ($25$ seconds forward simulation).
- **Multi-Step Training Loss:** Unrolled 3-step loss during training with scheduled sampling ensures stable recursive forecasting across $t+1 \dots t+5$.

---

## 7. Model Architectural Notes & Disclosures


1. **Stage Mapping:** Network traffic datasets (including CIC-IDS2018) do not provide native MITRE ATT&CK labels. CyberForecaster uses a transparent, expert-defined mapping layer (`preprocessing/stage_mapper.py`) mapping attack labels to **ATT&CK-aligned behavioural stages**.
2. **GNN Rollout Limitation:** The Temporal GNN model encodes dynamic network host graphs $G(t-9)\dots G(t)$ to predict future state vectors $S(t+1)\dots S(t+K)$. It does not synthesize future graph topology; the latest observed graph embedding $g(t)$ is carried forward as a proxy.
3. **Explainability:** Real-time SOC dashboard explanations use **Gradient Saliency Attribution** (PyTorch integrated gradients). Offline batch SHAP explanations are provided via `SHAPOfflineExplainer`.

---

## 8. Installation & Execution

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Run demo dataset generator
py data/demo_generator.py

# 3. Train models
py training/train_world_model.py
py training/train_baseline.py
py training/train_temporal_gnn.py

# 4. Run 4-model comparison benchmark
py training/compare_models.py

# 5. Launch Streamlit SOC Dashboard
streamlit run dashboard/app.py

# 6. Run test suite
py -m pytest
```
