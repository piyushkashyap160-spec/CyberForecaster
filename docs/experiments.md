# Experimental Evaluation & Comparative Benchmarks: CyberForecaster

## 1. Experimental Setup

The experimental framework compares three distinct defensive architectures on the exact same dataset, chronological split, and evaluation protocol:

1. **Model 1: Logistic Regression Baseline** — Non-temporal static feature classification baseline.
2. **Model 2: Temporal LSTM World Model** — Primary 23-dimensional state transition world model.
3. **Model 3: Temporal GNN + LSTM World Model** — Advanced graph-topology + temporal evolution world model.

---

## 2. Dataset & Split Configuration

- **Dataset:** Synthetic Demo Dataset (interleaved multi-scenario timeline across 1200 seconds).
- **Sequence Length:** $L = 10$ historical windows ($50$ seconds context).
- **Forecast Horizon:** $K = 5$ steps ($25$ seconds forward simulation).
- **Chronological Split (70/15/15):**
  - **Train (70%):** 161 sequence samples
  - **Validation (15%):** 34 sequence samples
  - **Test (15%):** 35 sequence samples (30 Benign / 85.7%, 5 Attack / 14.3%)

---

## 3. Comparative Benchmark Results

Machine-readable results saved in `experiments/results/benchmark_comparison.json`:

| Model Architecture | Precision | Recall | F1-Score | FPR | Next-State MAE | Next-State RMSE |
|---|---|---|---|---|---|---|
| **Logistic Regression Baseline** | 0.8333 | 1.0000 | 0.9091 | 0.0333 | N/A | N/A |
| **Temporal LSTM World Model** | **0.8333** | **1.0000** | **0.9091** | **0.0333** | **1.0376** | **6.4530** |
| **Temporal GNN + LSTM World Model** | **0.8333** | **1.0000** | **0.9091** | **0.0333** | **1.0850** | **6.6120** |

---

## 5. Canonical Benchmark on Real CSE-CIC-IDS2018 Dataset

The official canonical benchmark for CyberForecaster is evaluated on the full continuous stream of **CSE-CIC-IDS2018** (`Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`, $1,048,575$ total flows, $100,000$ chronological stream sample, $8,640$ five-second state windows).

### Canonical Specification:
- **Dataset:** CSE-CIC-IDS2018 (`Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`)
- **Evaluation Split:** Chronological zero-leakage split ($N=4,270$ Train, $N=1,708$ Val, $N=2,562$ Test)
- **Test Set Composition:** $N=859$ Benign sequences (33.5%), $N=1,703$ Malicious sequences (66.5%)
- **Sequence Length:** $L = 10$ windows ($50$ seconds context)
- **Forecast Horizon:** $K = 5$ steps ($25$ seconds forward simulation)
- **Feature Dimension:** 23-D flow-only state vector
- **Scaler:** `StateScaler` fitted strictly on Train split only (`models_weights/scaler.joblib`)
- **Model Checkpoint:** `models_weights/lstm_world_model.pt`
- **Random Seed:** 42
- **Decision Threshold:** $\theta = 0.90$ (selected on validation split)
- **Canonical Metrics:**
  - Precision: **0.9692** (96.92%)
  - Recall: **0.9965** (99.65%)
  - F1-Score: **0.9826** (98.26%)
  - False Positive Rate (FPR): **0.0629** (6.29%)
  - Confusion Matrix: TP = 1697, FP = 54, TN = 805, FN = 6
  - Multi-Step Future State RMSE ($K=1$): **2.0549** (vs. Naive Persistence: **11.7373**, Training-Mean: **2.1498**)
  - Validated Lead Time: **0.0s (Exact-Onset Detection)**

### Reproduction Command:
```bash
python training/run_strict_leakage_free_benchmark.py
```
Canonical artifact: `experiments/results/canonical_benchmark_results.json` and `models_weights/canonical_benchmark_results.json`.

---

## 6. Phase 2: Scientific Future-State Forecast Evaluation

**Research Question:**
*"Given the previous 50 seconds of network state (10 five-second windows), how accurately can CyberForecaster predict the network state 5, 15, and 25 seconds into the future, compared to simple persistence and training-mean baselines?"*

**Evaluation Standard:** Zero Temporal Leakage, Train-Fitted Scaler, Untouched Test Partition ($N=2,550$ multi-step sequences; $855$ Benign, $1,695$ Malicious).

### Multi-Horizon State Forecasting Performance Table

| Horizon | Prediction Advance | Model Architecture | RMSE (Scaled) | MAE (Scaled) | Relative Gain vs. Persistence | Relative Gain vs. Train Mean |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **$K=1$** | $+5\text{s}$ ahead | **Naive Persistence** | 5.0995 | 0.5445 | Baseline | — |
| | | **Training Mean** | 3.9613 | 0.7785 | Baseline | — |
| | | **Temporal LSTM World Model** | **3.9821** | 0.9523 | **+21.91%** | -0.53% |
| **$K=3$** | $+15\text{s}$ ahead | **Naive Persistence** | 5.3262 | 0.5889 | Baseline | — |
| | | **Training Mean** | 4.0120 | 0.7838 | Baseline | — |
| | | **Temporal LSTM World Model** | **4.1103** | 1.1898 | **+22.83%** | -2.45% |
| **$K=5$** | $+25\text{s}$ ahead | **Naive Persistence** | 5.3038 | 0.6107 | Baseline | — |
| | | **Training Mean** | 4.0222 | 0.7862 | Baseline | — |
| | | **Temporal LSTM World Model** | **4.1264** | 1.2416 | **+22.20%** | -2.59% |

### Statistical Robustness & Paired Hypothesis Testing (1,000 Bootstrap Resamples)
- **$K=1$ (+5s):** LSTM error reduction vs Persistence: Mean difference $= -0.2704$, 95% Bootstrap CI: $[-0.4037, -0.1354]$, paired $t = -3.98$ ($p < 0.0001$).
- **$K=3$ (+15s):** LSTM error reduction vs Persistence: Mean difference $= -0.4285$, 95% Bootstrap CI: $[-0.5670, -0.2945]$, paired $t = -6.18$ ($p < 0.0001$).
- **$K=5$ (+25s):** LSTM error reduction vs Persistence: Mean difference $= -0.4431$, 95% Bootstrap CI: $[-0.5841, -0.3052]$, paired $t = -6.21$ ($p < 0.0001$).

### Future-Aligned Attack Classification (Horizon $t+K$)
- **$K=1$ (+5s):** Precision $= 0.9984$, Recall $= 0.5327$, $F_1 = 0.6948$, $\text{FPR} = 0.0012$
- **$K=3$ (+15s):** Precision $= 0.9991$, Recall $= 0.6631$, $F_1 = 0.7971$, $\text{FPR} = 0.0012$
- **$K=5$ (+25s):** Precision $= 0.9991$, Recall $= 0.6295$, $F_1 = 0.7723$, $\text{FPR} = 0.0012$
*Scientific Note:* Because attacks in CSE-CIC-IDS2018 operate in long multi-hour burst episodes, future-aligned classification reflects the temporal persistence of malicious kill-chain states rather than isolated pre-onset early foresight.

### Validated Early-Warning Lead Time
- **Validated Pre-Onset Lead Time:** **0.0 seconds** (Exact-Onset Detection).
- **Candidate Gap Transition:** A single candidate detection triggered during the 25-second inter-burst gap between episodes due to lingering memory of preceding attack states; this is properly classified as an inter-burst transition rather than unvalidated advance foresight.

### Phase 2 Evaluation Command & Artifacts
```bash
python experiments/evaluate_future_forecast.py
```
- **Machine-readable Metrics:** `experiments/results/phase2_forecast_evaluation.json`
- **Forecast-vs-Reality Data:** `experiments/results/forecast_vs_reality.json` & `.csv`
- **Offline Curves Plot:** `experiments/results/forecast_vs_reality_curves.png`

---

## 7. Phase 3: Evidence-Grounded Forecast Explainability

### Core Research Question
*"Why did CyberForecaster make this forecast?"*

Phase 3 implements an evidence-grounded explainability architecture connecting 8 verifiable dimensions without LLM hallucination or fabricated telemetry:
1. **Model Evidence:** Real-time Gradient Saliency Attribution ($\frac{\partial P(\text{Attack})}{\partial S_{\tau, f}}$) evaluated across all 10 temporal sequence windows ($t-45\text{s} \dots t$) with explicit attribution targets and relative timestep indicators (`@ t-5s`, `@ t-10s`).
2. **Network Telemetry Evidence:** 23-D physical features vs recent-history baselines, reporting percentage deviation and elevation status (`elevated`, `suppressed`, `nominal`).
3. **Security Sensor Evidence:** Strict distinction between FastFlowDetector (Bot-vs-Benign per-flow classifier suspicion, not proof of attack) and Snort (5-tuple and timestamp-correlated signature alerts).
4. **Attack Path Evidence:** Correlated multi-hop progression chains ($A \to B \to C$) from the real-time reconstruction engine.
5. **Stage Transition Dynamics:** Mitre ATT&CK stage progression ($S_{t-1} \to S_t$) reported only when historical state genuinely exists.
6. **Predictive Uncertainty:** 10-sample Monte Carlo Dropout (`forward_with_mc_dropout`) yielding empirical predictive standard deviation and confidence bands, rigorously decoupled from point threat probabilities.
7. **Blockchain & Audit Provenance:** SHA-256 data hash and transaction verification on the Hardhat ledger.
8. **Scientific Limitations:** Explicit disclosures regarding the 0.0s pre-onset benchmark lead time.

### Key Conceptual Distinctions
- **Threat Probability vs. Predictive Uncertainty:** Threat Probability is the point estimate $P(\text{Attack} \mid S_{t-9 \dots t}) \in [0, 1]$. Predictive Uncertainty is the empirical dispersion (standard deviation) across stochastic dropout perturbations quantifying epistemic model uncertainty.
- **Future-State Forecast vs. Early-Warning Lead Time:** A future-state forecast predicts the multi-step state trajectory $\hat{S}_{t+K}$ and attack risk forward in time. Early-warning lead time measures the advance interval between detection and ground-truth attack onset. On the CSE-CIC-IDS2018 benchmark, the validated pre-onset lead time is 0.0s (exact-onset detection).
- **Model Attribution vs. Sensor Evidence:** Gradient saliency identifies which input dimensions influenced the neural network's loss gradient. Sensor evidence (Snort rules, FastFlowDetector anomalies) represents external ground-truth detections from signature and flow inspection engines.

### Canonical Explanation Payload Example
```json
{
  "forecast_id": "4b726488-829d-476c-8f1b-3efb7c63db44",
  "timestamp": "2026-09-07T20:15:05.123456+00:00",
  "host_ip": "192.168.1.15",
  "forecast": {
    "attack_probability": 0.91,
    "threat_probability": 0.91,
    "predicted_stage": "Lateral Movement",
    "forecast_horizon_seconds": 25,
    "severity": "CRITICAL"
  },
  "narrative_summary": "CyberForecaster projected Lateral Movement for host 192.168.1.15 with a Threat Probability of 91%. Gradient saliency identifies syn_ratio (t-5s), unique_dst_ports (t-10s) as primary model risk drivers. Observed network telemetry shows elevated SYN Flag Ratio (+812.3%) against recent baseline. A temporally correlated Snort alert (SID 1000003: COMMUNITY WEB-MISC Lateral Movement RPC probe) was verified. FastFlowDetector flagged the corresponding flow with suspicion score 0.94. Active multi-hop attack path path-192.168.1.10-192.168.1.24-17881 was reconstructed across 2 hops. Forecast indicates a transition from Reconnaissance toward Lateral Movement. Predictive uncertainty was estimated at std=0.034 across 10 MC-dropout passes.",
  "model_evidence": {
    "method": "Gradient Saliency Attribution (Fast Real-Time)",
    "target": "attack_probability",
    "top_temporal_features": [
      {
        "feature": "syn_ratio",
        "timestep": "t-5s",
        "label": "syn_ratio @ t-5s",
        "attribution": 0.38421,
        "direction": "increases_risk"
      },
      {
        "feature": "unique_dst_ports",
        "timestep": "t-10s",
        "label": "unique_dst_ports @ t-10s",
        "attribution": 0.29104,
        "direction": "increases_risk"
      }
    ]
  },
  "telemetry_evidence": [
    {
      "feature": "syn_ratio",
      "feature_label": "SYN Flag Ratio",
      "current_value": 0.92,
      "baseline_value": 0.10,
      "deviation_pct": 820.0,
      "direction": "elevated",
      "source": "NetworkState"
    }
  ],
  "sensor_evidence": {
    "fastflow": {
      "available": true,
      "score": 0.94,
      "label": "Bot / Suspicious",
      "disclaimer": "FastFlowDetector provides flow anomaly suspicion and does not independently constitute proof of attack."
    },
    "snort": {
      "available": true,
      "sid": "1000003",
      "message": "COMMUNITY WEB-MISC Lateral Movement RPC probe",
      "timestamp": "2026-09-07T20:15:00Z"
    }
  },
  "stage_transition": {
    "previous_stage": "Reconnaissance",
    "predicted_stage": "Lateral Movement",
    "transition_detected": true,
    "summary": "Forecast indicates a transition from Reconnaissance toward Lateral Movement."
  },
  "attack_path": {
    "available": true,
    "path_id": "path-192.168.1.10-192.168.1.24-17881",
    "hop_count": 2,
    "severity": "HIGH",
    "summary": "Forecast is associated with an observed multi-hop progression (192.168.1.10 -> 192.168.1.24, 2 hops)."
  },
  "uncertainty": {
    "available": true,
    "method": "MC-Dropout (10 stochastic forward passes)",
    "mean_probability": 0.908,
    "std": 0.034,
    "confidence_band": [0.841, 0.975]
  },
  "provenance": {
    "model": "TemporalLSTMWorldModel (23-D, 128 hidden, 2 layers)",
    "scaler": "Train-Fitted StateScaler",
    "data_hash": "d83b54a85c8e1e7fa890e0b3c678a1",
    "blockchain_registered": true,
    "blockchain_tx": "0x7b58c93a8e9d1234abcd",
    "evidence_sources": [
      "TemporalLSTM",
      "NetworkState",
      "FastFlowDetector",
      "Snort",
      "AttackPathReconstructor",
      "MC-Dropout",
      "BlockchainLedger"
    ]
  },
  "limitations": [
    "Canonical CSE-CIC-IDS2018 benchmark evaluates future-state trajectory with 0.0s validated pre-onset lead time (exact-onset detection).",
    "FastFlowDetector provides per-flow anomaly suspicion and does not independently constitute proof of attack.",
    "MC-Dropout provides an empirical model uncertainty band and is not a certified statistical confidence interval."
  ]
}
```
