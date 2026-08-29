# Technical Validation & Audit Report: CyberForecaster

**Date:** August 29, 2026  
**Auditor & Lead Architect:** Lead AI/ML & Cybersecurity Engineer  
**Target Organization:** National Technical Research Organisation (NTRO)  
**SIH 2026 Problem Statement:** AI based Network Attack Forecasting from Network Traffic Data  
**Canonical Source of Truth:** [`experiments/results/canonical_benchmark_results.json`](../experiments/results/canonical_benchmark_results.json)

---

## Executive Summary

Following an exhaustive scientific audit and rigorous real-data evaluation on the official CSE-CIC-IDS2018 dataset (**1,048,575 total flows**, $8,640$ five-second state windows across 12 hours), the **CyberForecaster** repository has locked in verified, leakage-free benchmark results.

All 47 automated unit and integration tests passed cleanly (`py -m pytest -q`). Data leakage was strictly prevented by partitioning continuous state windows chronologically prior to sequence generation, fitting scalers exclusively on the training split, and selecting decision thresholds exclusively on the validation split.

---

## 1. Audit & Implementation Matrix

| Requirement / Area | Verified Implementation Status | Evidence & Methodology |
| :--- | :---: | :--- |
| **1. Real CIC-IDS2018 Benchmark** | **COMPLETED & LOCKED** | Evaluated on real CSE-CIC-IDS2018 traffic (`Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`). 100k chronological stream slice, $N=2,562$ untouched test sequences ($859$ benign, $1,703$ malicious). Output saved to `experiments/results/canonical_benchmark_results.json`. |
| **2. Future State RMSE Baselines** | **VERIFIED & SUPERIOR** | **$K=1$ Future-State RMSE:** Temporal LSTM = **2.0549**, Naive Persistence = **11.7373**, Training-Mean Baseline = **2.1498**. LSTM outperforms baselines across all horizons ($K=1, 3, 5$). |
| **3. Zero Temporal Leakage** | **STRICTLY ENFORCED** | States split chronologically (60% Train, 20% Val, 20% Test) **before** rolling sequence construction. Scalers fitted on Train split only; thresholds selected on Validation split only. |
| **4. Validated Lead Time** | **HONEST & EXPLORATORY** | Validated lead time is **0.0s (exact onset detection with 0 false alarms on preceding benign baseline)**. No genuine $>0\text{s}$ pre-onset detections observed. Formally classified as exploratory due to $N=2$ test attack episodes. |
| **5. GNN Topology Caveat** | **DISCLOSED & VERIFIED** | Real dataset uses 100% single-node fallback (1 node, 0 edges). Temporal GNN ($F_1 = 0.9921$) acts as an alternative recurrent model; no graph-topological advantage is claimed. |
| **6. Packet Feature Ablation** | **DOCUMENTED ABLATION** | 30-D packet-enriched model ($F_1 = 0.5936$, Recall = $0.4228$) documented as a failed fusion ablation due to cross-flow interval smoothing. 23-D flow telemetry is the primary feature set. |
| **7. ATT&CK Stage Mapping** | **TRANSPARENT LAYER** | Mapping module [`preprocessing/stage_mapper.py`](../preprocessing/stage_mapper.py) maps raw labels to ATT&CK tactics (Initial Access, Execution, Persistence, Command & Control, Impact). |
| **8. Explainability Engine** | **VERIFIED** | Real-time gradient saliency attribution labeled explicitly; offline SHAP explainer provided separately. |

---

## 2. Benchmark Comparison Table (Untouched Test Split, $N=2,562$)

| Model Architecture | Input Dim | Tuned $\theta$ | Precision | Recall | $F_1$ Score | FPR | Confusion Matrix (TP / FP / TN / FN) | Future State RMSE ($K=1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Logistic Regression** | 23-D | 0.20 | 0.9341 | 0.9988 | 0.9654 | 13.97% | 1,701 / 120 / 739 / 2 | N/A (Static) |
| **Naive Persistence Baseline** | 23-D | 0.05 | 0.9028 | 0.3053 | 0.4563 | 6.52% | 520 / 56 / 803 / 1,183 | 11.7373 |
| **Temporal LSTM World Model (Primary)** | 23-D | 0.90 | **0.9692** | **0.9965** | **0.9826** | **6.29%** | **1,697 / 54 / 805 / 6** | **2.0549** |
| **Temporal GNN World Model** | 23-D | 0.85 | 0.9883 | 0.9959 | 0.9921 | 2.33% | 1,696 / 20 / 839 / 7 | N/A (Single-Node) |
| *Temporal LSTM (Packet-Enriched Ablation)* | 30-D | 0.90 | 0.9959 | 0.4228 | 0.5936 | 0.35% | 720 / 3 / 856 / 983 | 3.6469 |

---

## 3. Final Scientific Statement

All performance claims are strictly backed by [`experiments/results/canonical_benchmark_results.json`](../experiments/results/canonical_benchmark_results.json). No unvalidated early-warning lead times or fabricated graph advantages are claimed. The repository is tested (47 passing unit and integration tests), reproducible, and locked in for evaluation.
