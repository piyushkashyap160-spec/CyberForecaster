# CyberForecaster: AI-Based Network Attack Forecasting System

> **Tagline:** *From Intrusion Detection to Attack Forecasting*  
> **SIH 2026 Problem Statement:** AI based Network Attack Forecasting from Network Traffic Data  
> **Target Organization:** National Technical Research Organisation (NTRO)

---

## 1. Project Overview

**CyberForecaster** is an open-source, offline AI prototype engineered to model temporal network state dynamics and forecast cyberattacks before initial compromise is completed. Designed for Critical Information Infrastructure (CII) and enterprise Security Operations Centers (SOC), CyberForecaster shifts cybersecurity defense from reactive intrusion detection to proactive attack forecasting.

---

## 2. Problem Statement

Modern nation-state cyberattacks and Advanced Persistent Threats (APTs) operate via multi-stage kill chains. Traditional intrusion detection systems (IDS) trigger alerts only after malicious payloads or unauthorized access events have already occurred. NTRO's SIH 2026 problem statement requires an open-source AI system that learns network behavior, models temporal state dynamics, predicts future network states, and maps forecasted activity to MITRE ATT&CK stages with interpretable explanations.

---

## 3. Proposed Solution: The World Model Approach

CyberForecaster replaces static classification with a **Temporal World Model** learning state dynamics:

$$ P(S[t+1] \mid S[t], S[t-1], \dots, S[t-L+1]) $$

```text
CyberForecaster Paradigm:
Traffic History
      ↓
Current Network State S(t)
      ↓
Learn Temporal Transition Dynamics
      ↓
Predict Next State S(t+1)
      ↓
Recursive K-Step Forward Rollout [S(t+1) ... S(t+K)]
      ↓
Forecast Future Attack Probability & MITRE ATT&CK Stage
      ↓
Explainable Early Warning Threshold Alert
```

---

## 4. Benchmark Evaluation & Statistical Performance

> [!IMPORTANT]
> **Dataset Disclaimer:** Results below are evaluated on an interleaved multi-scenario demo dataset (35 test sequences: 30 Benign / 85.7%, 5 Attack / 14.3%). Real-world evaluation pipeline (`py data/ingest_cicids2018.py`) is provided for processing raw CIC-IDS2018 CSV datasets dropped into `data/raw/`.

### Binary Attack Forecast Benchmarks

| Metric | Baseline (Logistic Regression) | CyberForecaster (Temporal World Model) |
|---|---|---|
| **Precision** | 0.8333 | **0.8333** |
| **Recall** | 1.0000 | **1.0000** |
| **F1-Score** | 0.9091 | **0.9091** |
| **False Positive Rate (FPR)** | 0.0333 (3.33%) | **0.0333 (3.33%)** |
| **Confusion Matrix (TP/FP/TN/FN)** | 5 / 1 / 29 / 0 | **5 / 1 / 29 / 0** |
| **Next-State MAE** | N/A | **1.0376** |
| **Next-State MSE** | N/A | **41.6411** |
| **Next-State RMSE** | N/A | **6.4530** |

---

## 5. Horizon-Wise K-Step Forecasting Performance

- **Sequence Length:** $L = 10$ historical windows ($50$ seconds context).
- **Forecast Horizon:** $K = 5$ steps ($25$ seconds forward simulation).

Uncertainty propagation across forward simulation steps:

| Horizon Step | Precision | Recall | F1-Score | False Positive Rate (FPR) |
|---|---|---|---|---|
| **$t+1$** | 0.6667 | 1.0000 | 0.8000 | 0.0667 (6.67%) |
| **$t+2$** | 0.5000 | 1.0000 | 0.6667 | 0.1000 (10.00%) |
| **$t+3$** | 0.2857 | 1.0000 | 0.4444 | 0.1667 (16.67%) |
| **$t+4$** | 0.1429 | 1.0000 | 0.2500 | 0.2000 (20.00%) |
| **$t+5$** | 0.0000 | 0.0000 | 0.0000 | 0.2333 (23.33%) |

---

## 6. Installation & Execution

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Run demo dataset generator
py data/demo_generator.py

# 3. Train models
py training/train_world_model.py
py training/train_baseline.py

# 4. Run evaluation
py training/evaluate.py

# 5. Run test suite
py -m pytest

# 6. Launch Streamlit SOC Dashboard
streamlit run dashboard/app.py
```
