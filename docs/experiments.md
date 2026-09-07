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
