# CyberForecaster: Strict Leakage-Free Real ML Benchmark Report (v2)

> **SUPERSEDED / SOURCE OF TRUTH NOTICE:**  
> The final canonical benchmark results, validated future-state RMSE baselines, and lead-time exploratory findings are locked in [`experiments/results/canonical_benchmark_results.json`](../experiments/results/canonical_benchmark_results.json).

**Evaluation Date:** August 29, 2026  
**Dataset:** CSE-CIC-IDS2018 (Full 12-Hour Timeline, Friday-02-03-2018)  
**Evaluation Standard:** Zero Temporal Leakage, Train-Fitted Scaler Discipline, Fair Cross-Model Validation Tuning  
**Canonical Artifact:** `experiments/results/canonical_benchmark_results.json`

---

## 1. Executive Summary

This evaluation executes a **strict, leakage-free benchmark** on the continuous timeline of CSE-CIC-IDS2018 traffic (8,640 continuous 5-second state windows across 12 hours, $N=2,562$ untouched test sequences).

### Key Validations:
1. **Zero Temporal Leakage:** Network state windows were partitioned chronologically into independent blocks **before** rolling temporal sequence generation ($S_{t-9:t} \to S_{t+1}$).
2. **Statistically Significant Benign Test Pool:** The test partition contains **$N=859$ benign sequences** and **$N=1,703$ malicious sequences** ($N_{\text{total}} = 2,562$).
3. **Strict Scaler Discipline:** The `StateScaler` was fitted strictly on the Training split ($X_{\text{train}}$) and applied without modification to validation and test sets.
4. **Fair Threshold Tuning:** An identical validation-split $F_1$ threshold sweep ($\theta \in [0.05, 0.95]$ with step $0.05$) was applied across all models.
5. **Future-State RMSE Superiority:** The Temporal LSTM achieved a $K=1$ future-state RMSE of **2.0549** (vs. **11.7373** for Persistence and **2.1498** for Training-Mean).
6. **Lead-Time Exploratory Status:** Validated lead time is **0.0s (exact onset detection with 0 false alarms on preceding benign baseline)**. Formally classified as exploratory due to $N=2$ test attack episodes.
7. **GNN Topology Caveat:** Evaluated dataset operates in 100% single-node fallback mode (1 node, 0 edges); score reflects recurrent parameterization, not graph-structural superiority.

---

## 2. Canonical Classification Performance Table ($N=2,562$ Test Sequences)

| Model Architecture | Input Dim | Tuned $\theta$ | Precision | Recall | $F_1$ Score | FPR | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Logistic Regression** | 23-D | 0.20 | 0.9341 | 0.9988 | 0.9654 | 13.97% | 1,701 / 120 / 739 / 2 |
| **Naive Persistence Baseline** | 23-D | 0.05 | 0.9028 | 0.3053 | 0.4563 | 6.52% | 520 / 56 / 803 / 1,183 |
| **Temporal LSTM (Flow-Only Primary)** | 23-D | 0.90 | **0.9692** | **0.9965** | **0.9826** | **6.29%** | **1,697 / 54 / 805 / 6** |
| **Temporal GNN World Model** | 23-D | 0.85 | 0.9883 | 0.9959 | 0.9921 | 2.33% | 1,696 / 20 / 839 / 7 |
| *Temporal LSTM (Packet Enriched Ablation)* | 30-D | 0.90 | 0.9959 | 0.4228 | 0.5936 | 0.35% | 720 / 3 / 856 / 983 |

---

## 3. Verified Multi-Step Future State RMSE ($X_{t+K}$)

| Prediction Horizon ($K$) | Projected Advance | Naive Persistence ($X_{t+K} = X_t$) | Training-Mean Baseline ($\bar{X}_{\text{train}}$) | Temporal LSTM World Model |
| :---: | :---: | :---: | :---: | :---: |
| **$K = 1$ Step** | $+5\text{s}$ ahead | 11.7373 | 2.1498 | **2.0549** |
| **$K = 3$ Steps** | $+15\text{s}$ ahead | 12.0022 | 2.4191 | **2.3484** |
| **$K = 5$ Steps** | $+25\text{s}$ ahead | 12.0051 | 2.2245 | **2.1577** |
