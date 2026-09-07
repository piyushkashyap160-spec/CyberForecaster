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
