# CyberForecaster: Real-Data Gap Analysis & Lead-Time Audit Report

> **CANONICAL SOURCE OF TRUTH NOTICE:**  
> The final canonical benchmark results, validated future-state RMSE baselines, and lead-time exploratory findings are locked in [`experiments/results/canonical_benchmark_results.json`](../experiments/results/canonical_benchmark_results.json).

**Date:** August 29, 2026  
**Dataset:** CSE-CIC-IDS2018 (Friday-02-03-2018, 100,000 flows, 8,640 5-second state windows)  
**Evaluation Standard:** Zero Temporal Leakage, Train-Fitted Scalers, Multi-Episode Chronological Audit  

---

## 1. Executive Summary: The Lead-Time Investigation

A critical audit of the lead-time metric was conducted to investigate why the static Logistic Regression (LR) baseline initially reported $+50.0\text{s}$ mean lead time compared to $+25.0\text{s}$ for Temporal LSTM and Temporal GNN.

### Core Finding
**Static LR's $+50.0\text{s}$ lead time on the primary attack onset is a mathematical artifact of False-Positive Pre-Alarming, NOT legitimate early detection.**

Because Static LR operates at a high False Positive Rate ($\text{FPR} = 13.97\%$, $\theta = 0.20$), its predictions randomly fire positive alarms during normal benign traffic. The search window in `forecasting/lead_time.py` scans backwards up to 10 windows ($50\text{s}$) prior to ground-truth attack onset; when an uncorrected false alarm occurs in that pre-attack window, the algorithm credits the model with the maximum possible $+50.0\text{s}$ early warning.

---

## 2. Per-Episode Lead-Time Trace & Classification

The test partition contains **two distinct attack episodes**:
- **Episode 0:** Onset at `2018-03-02 10:36:50` (index 854), End at `12:13:05` (index 1999). Duration: $5,730\text{s}$ ($1,146$ windows).
- **Episode 1:** Onset at `2018-03-02 12:13:35` (index 2005), End at `12:59:55` (index 2561). Duration: $2,785\text{s}$ ($557$ windows).

### Per-Model Episode Audit Breakdown

| Model | Tuned $\theta$ | Ep 0 Onset | Ep 0 Trigger | Ep 0 Raw Lead | Ep 0 Classification | Ep 1 Onset | Ep 1 Trigger | Ep 1 Raw Lead | Ep 1 Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static LR** | 0.20 | 10:36:50 | 09:23:10 | **+50.0s** | **False-Positive Pre-Alarm (Class B)** | 12:13:35 | 12:12:45 | **+50.0s** | True Early Warning (Class A) |
| **Naive Persistence** | 0.05 | 10:36:50 | 09:23:45 | **+15.0s** | **False-Positive Pre-Alarm (Class B)** | 12:13:35 | 12:12:45 | **+50.0s** | True Early Warning (Class A) |
| **Temporal LSTM (23-D)** | 0.90 | 10:36:50 | 10:36:50 | **0.0s** | **Post-Onset Detection (Class C)** | 12:13:35 | 12:12:45 | **+50.0s** | True Early Warning (Class A) |
| **Temporal GNN (23-D)** | 0.85 | 10:36:50 | 09:23:55 | **+5.0s** | **False-Positive Pre-Alarm (Class B)** | 12:13:35 | 12:12:45 | **+50.0s** | True Early Warning (Class A) |
| **Temporal LSTM (30-D)** | 0.90 | 10:36:50 | 10:45:20 | **-510.0s** | **Delayed Detection (Class C)** | 12:13:35 | 12:12:50 | **+45.0s** | True Early Warning (Class A) |

### Classification Distribution
- **Static LR:** 50% False-Positive Pre-Alarms (Ep 0 triggered on benign data at 09:23:10), 50% True Early Warning (Ep 1 inter-burst transition).
- **Temporal LSTM (23-D):** **0% False-Positive Pre-Alarms**, 50% Exact Onset Detection, 50% True Early Warning. Temporal LSTM is the most disciplined model on benign baselines.

---

## 3. Raw vs. Validated Early-Warning Lead Time

When False-Positive Pre-Alarms (detections triggered on ground-truth benign states prior to attack onset) are excluded, the validated lead-time metrics become:

| Model Architecture | Raw Mean Lead Time | Validated Mean Lead Time (Excluding FP Pre-Alarms) | Median Lead Time | Max Lead Time | Episodes Detected / Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Logistic Regression** | +50.0s | **+50.0s** *(only 1 valid episode)* | +50.0s | +50.0s | 2 / 2 (1 FP pre-alarm) |
| **Naive Persistence Baseline** | +32.5s | **+50.0s** *(only 1 valid episode)* | +32.5s | +50.0s | 2 / 2 (1 FP pre-alarm) |
| **Temporal LSTM (23-D Flow-Only)** | **+25.0s** | **+25.0s** *(both episodes valid)* | **+25.0s** | **+50.0s** | **2 / 2 (0 FP pre-alarms)** |
| **Temporal GNN (23-D Graph)** | +27.5s | **+50.0s** *(only 1 valid episode)* | +27.5s | +50.0s | 2 / 2 (1 FP pre-alarm) |
| **Temporal LSTM (30-D Enriched)** | -232.5s | **-232.5s** *(both episodes valid)* | -232.5s | +45.0s | 2 / 2 (0 FP pre-alarms) |

---

## 4. 30-D Packet-Enriched Failure Diagnosis

The 30-D model reported poor recall ($0.4228$) and severe detection latency ($-510\text{s}$ on Episode 0).

### Sensitivity Analysis on Validation Set
Sweeping thresholds across $\theta \in [0.05, 0.90]$ on the independent validation split revealed:
- $\theta = 0.05 \to \text{Precision} = 0.9646, \text{Recall} = 0.6082, F_1 = 0.7460$
- $\theta = 0.50 \to \text{Precision} = 0.9779, \text{Recall} = 0.6082, F_1 = 0.7500$
- $\theta = 0.90 \to \text{Precision} = 0.9916, \text{Recall} = 0.6082, F_1 = 0.7540$

### Root Cause
Recall is capped at $\approx 60.8\%$ across all thresholds. The 7 PCAP-derived features (TTL variance, packet size variance, inter-arrival time variance, port entropy) are computed as global 5-second capture summaries. During high-throughput periods, benign background traffic smooths out packet-level statistical anomalies, diluting the subtle signatures of botnet C2 traffic. Flow-level statistical metrics ($23\text{-D}$) remain significantly more discriminative.

---

## 5. Multi-Step State RMSE vs. Baselines

Evaluating scaled state vector prediction errors on the test partition:

| Model / Baseline | $K=1$ ($+5\text{s}$) | $K=3$ ($+15\text{s}$) | $K=5$ ($+25\text{s}$) |
| :--- | :---: | :---: | :---: |
| **Training Mean Baseline ($\bar{X}_{\text{train}}$)** | 2.3248 | 2.3248 | 2.3248 |
| **Temporal LSTM World Model** | **2.2567** | **2.2615** | **2.2634** |

The LSTM World Model outperforms the static training-mean baseline across all horizons, maintaining stable, non-diverging error growth.

---

## 6. Real GNN Graph Topology Truth

An empirical inspection of all 8,640 state windows in `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` revealed:
- **Average Nodes / Window:** $1.00$ ($\text{Median} = 1.0, \text{Max} = 1$)
- **Average Edges / Window:** $0.00$ ($\text{Median} = 0.0, \text{Max} = 0$)
- **Percentage Single-Node Fallback:** **100.00%**

### Scientific Conclusion
Because the CSE-CIC-IDS2018 CSV distribution provides pre-aggregated interface flows without host IP pairs, the graph encoder operates in **100% single-node fallback mode**. Consequently, the GNN's $F_1 = 0.9921$ score does **not** provide empirical evidence of topological graph superiority over LSTM; rather, it behaves as an alternate recurrent parameterization on the same temporal vectors.

---

## 7. Data Sampling & Chronological Leakage Verification

- **Total CSV Flows:** $1,048,575$
- **Selected Flows:** $100,000$ (chronological stream slice, preserving raw order)
- **Timestamp Coverage:** `2018-03-02 01:00:00` to `2018-03-02 12:59:59` ($12$ continuous hours)
- **Temporal Gaps (>5s):** $0$
- **Leakage Guarantees:**
  - $\max(\text{Train TS}) < \min(\text{Val TS})$: Confirmed across independent chronological block partitions.
  - $\max(\text{Val TS}) < \min(\text{Test TS})$: Confirmed.
  - No sequence crosses partition boundaries.
  - Scaler fitted strictly on Training split.
  - Threshold selected strictly on Validation split.
