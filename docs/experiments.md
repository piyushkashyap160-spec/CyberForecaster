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

## 4. Horizon-Wise Forecasting Decay ($t+1 \dots t+5$)

Tracking uncertainty propagation over forward rollout steps:

| Forecast Step | Precision | Recall | F1-Score | FPR |
|---|---|---|---|---|
| **$t+1$** | 0.6667 | 1.0000 | 0.8000 | 0.0667 |
| **$t+2$** | 0.5000 | 1.0000 | 0.6667 | 0.1000 |
| **$t+3$** | 0.2857 | 1.0000 | 0.4444 | 0.1667 |
| **$t+4$** | 0.1429 | 1.0000 | 0.2500 | 0.2000 |
| **$t+5$** | 0.0000 | 0.0000 | 0.0000 | 0.2333 |
