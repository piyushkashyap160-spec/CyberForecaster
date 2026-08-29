# CyberForecaster: AI-Based Network Attack Forecasting System

> **Tagline:** *From Intrusion Detection to Attack Forecasting*  
> **SIH 2026 Problem Statement:** AI based Network Attack Forecasting from Network Traffic Data  
> **Target Organization:** National Technical Research Organisation (NTRO)  
> **Canonical Benchmark:** [`experiments/results/canonical_benchmark_results.json`](experiments/results/canonical_benchmark_results.json)

---

## 1. Project Overview

**CyberForecaster** is an open-source, offline AI system engineered to model temporal network state dynamics and forecast cyberattacks before compromise is completed. Designed for Critical Information Infrastructure (CII) and Security Operations Centers (SOC), CyberForecaster shifts cybersecurity defense from reactive signature matching to proactive future-state trajectory forecasting.

---

## 2. Problem Statement & Paradigm

Modern nation-state cyberattacks operate via multi-stage kill chains. Traditional intrusion detection systems (IDS) trigger alerts only after malicious packets match static signatures. CyberForecaster models continuous network telemetry as an autoregressive **Temporal World Model**:

$$ P(S_{t+1} \mid S_t, S_{t-1}, \dots, S_{t-L+1}) $$

```text
CyberForecaster Architecture:
Telemetry Stream History [S(t-9) ... S(t)]
       ↓
Current Network State S(t)
       ↓
Temporal Transition Dynamics (LSTM / Recurrent Cell)
       ↓
Future State Forecast S(t+1)
       ↓
Recursive K-Step Forward Rollout [S(t+1) ... S(t+K)]
       ↓
Forecasted Threat Probability & ATT&CK-Aligned Stage
       ↓
Uncertainty-Aware Alerting (Monte Carlo Dropout & Saliency)
```

---

## 3. Real CSE-CIC-IDS2018 Benchmark Performance

The canonical benchmark was evaluated across the full multi-hour stream of `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` ($1,048,575$ total flows, $100,000$ chronological stream sample, $8,640$ 5-second state windows). Zero temporal leakage was enforced by partitioning states chronologically into independent blocks prior to sequence generation, fitting scalers strictly on the training partition, and tuning decision thresholds exclusively on the validation split.

### A. Classification Performance on Untouched Test Partition ($N=2,562$ Sequences; 859 Benign, 1,703 Malicious)

| Model Architecture | Feature Dim | Tuned Threshold ($\theta$) | Precision | Recall | $F_1$ Score | FPR | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Logistic Regression** | 23-D | 0.20 | 0.9341 | 0.9988 | 0.9654 | 13.97% | 1,701 / 120 / 739 / 2 |
| **Naive Persistence Baseline** | 23-D | 0.05 | 0.9028 | 0.3053 | 0.4563 | 6.52% | 520 / 56 / 803 / 1,183 |
| **Temporal LSTM World Model (Primary)** | 23-D | 0.90 | **0.9692** | **0.9965** | **0.9826** | **6.29%** | **1,697 / 54 / 805 / 6** |
| **Temporal GNN World Model** | 23-D | 0.85 | 0.9883 | 0.9959 | 0.9921 | 2.33% | 1,696 / 20 / 839 / 7 |
| *Temporal LSTM (Packet-Enriched Ablation)* | 30-D | 0.90 | 0.9959 | 0.4228 | 0.5936 | 0.35% | 720 / 3 / 856 / 983 |

*Source of truth: `experiments/results/canonical_benchmark_results.json`*

---

## 4. Multi-Step Future State Prediction ($X_{t+K}$ RMSE)

### Primary Scientific Finding:
On the evaluated real CIC-IDS2018 test data, the **Temporal LSTM achieved a $K=1$ future-state RMSE of 2.0549, compared with 11.7373 for persistence and 2.1498 for the training-mean baseline**, maintaining superior predictive accuracy across all evaluated forward simulation horizons:

| Prediction Horizon ($K$) | Projected Advance | Naive Persistence ($X_{t+K} = X_t$) | Training-Mean Baseline ($\bar{X}_{\text{train}}$) | Temporal LSTM World Model |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 1$ Step** | $+5\text{s}$ ahead | 11.7373 | 2.1498 | **2.0549** |
| **$K = 3$ Steps** | $+15\text{s}$ ahead | 12.0022 | 2.4191 | **2.3484** |
| **$K = 5$ Steps** | $+25\text{s}$ ahead | 12.0051 | 2.2245 | **2.1577** |

---

## 5. Lead-Time Evaluation & Exploratory Findings

### Secondary Finding:
On the evaluated test timeline, the **Temporal LSTM detected attacks at the exact onset window ($0.0\text{s}$ latency) with zero false alarms during the preceding benign baseline**. No genuine $>0\text{s}$ pre-onset detections were observed.

- **Exploratory Status:** Because the test block contains only **$N=2$ distinct attack episodes**, lead-time evaluation is strictly exploratory. A sample size of 2 episodes is insufficient for generalizable population lead-time claims.
- **False-Alarm Pre-Alarm Audit:** The static Logistic Regression baseline's apparent $+50.0\text{s}$ lead time was proven to be a mathematical artifact of its high false-positive rate ($13.97\%$), triggering random false alarms on benign traffic over an hour prior to attack onset.

---

## 6. Architectural Notes & Explicit Disclosures

1. **GNN Topology Caveat:** The evaluated CSE-CIC-IDS2018 dataset is pre-aggregated per interface without host IP pair topology. The Temporal GNN operated in **100% single-node fallback mode (1 node, 0 edges)**. Its classification score reflects recurrent parameterization, **not** demonstrated graph-structural superiority.
2. **Packet-Enriched Model Failure:** The 30-D packet-enriched model suffered degraded recall ($42.28\%$) because 5-second interval PCAP feature aggregation smoothed out subtle flow-level attack anomalies. Flow-level telemetry (23-D) remains the validated primary feature set.
3. **ATT&CK Stage Mapping:** Network traffic datasets do not include native MITRE ATT&CK labels. CyberForecaster maps attack classes to ATT&CK tactics via an expert-defined transparent mapping layer (`preprocessing/stage_mapper.py`).

---

## 7. Installation & Quickstart

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Run unit & regression test suite (47 tests)
py -m pytest -q

# 3. Inspect canonical benchmark results
py -c "import json; print(json.dumps(json.load(open('experiments/results/canonical_benchmark_results.json')), indent=2))"
```
