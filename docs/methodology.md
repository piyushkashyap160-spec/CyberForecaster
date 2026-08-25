# Mathematical & Scientific Methodology: CyberForecaster

## 1. Problem Formulation

Let $S(t) \in \mathbb{R}^{23}$ represent the network state vector observed at time window $t$. Given a sequence of $L$ historical network state observations:

$$ \mathbf{X}_t = [S(t-L+1), S(t-L+2), \dots, S(t)] $$

The goal of the Temporal World Model is to learn the conditional distribution over next network state $S(t+1)$, future attack probability $P(\text{Attack}_{t+1})$, and MITRE ATT&CK attack stage $Y_{\text{stage}, t+1}$:

$$ \hat{S}(t+1), \hat{P}(\text{Attack}_{t+1}), \hat{Y}_{\text{stage}, t+1} = f_\theta(\mathbf{X}_t) $$

---

## 2. Multi-Task Objective Function

The network parameters $\theta$ are optimized end-to-end using a weighted multi-task loss formulation:

$$ \mathcal{L}(\theta) = w_1 \mathcal{L}_{\text{MSE}}(\hat{S}_{t+1}, S_{t+1}) + w_2 \mathcal{L}_{\text{BCE}}(\hat{P}_{t+1}, Y_{t+1}) + w_3 \mathcal{L}_{\text{CE}}(\hat{Y}_{\text{stage}}, Y_{\text{stage}}) $$

Where:
- $w_1 = 1.0$, $w_2 = 0.5$, $w_3 = 0.5$
- $\mathcal{L}_{\text{MSE}}$ enforces physical state trajectory alignment in continuous feature space.
- $\mathcal{L}_{\text{BCE}}$ trains the attack risk head.
- $\mathcal{L}_{\text{CE}}$ aligns stage classification across MITRE ATT&CK stages.

---

## 3. Evaluation Split Methodology & Scientific Rigor

To prevent target and feature scaling data leakage:
1. **Scaler Fitting:** Standard scaling parameters are fitted strictly on the 70% training split (`X_train`) to ensure zero leakage into validation (15%) and test (15%) sets.
2. **Multi-Scenario Interleaved Timelines:** Evaluation timelines interleave benign recovery periods between attack phases (Recon, Initial Access, Lateral Movement, C2, Exfiltration). This guarantees that the chronological test split contains a realistic mix of benign (85.7%) and attack (14.3%) windows for measuring Precision, Recall, F1, and False Positive Rate (FPR).
3. **Synthetic Data Disclaimer:** Results obtained on demo data are clearly designated as *Synthetic Demonstration Results* and separated from real-world dataset pipelines (`data/ingest_cicids2018.py`).

---

## 4. Horizon-Wise K-Step Forecasting Evaluation

For forward forecasting over horizon $K$:
1. Set $\mathbf{X}^{(1)} = [S(t-L+1), \dots, S(t)]$.
2. For step $k = 1, \dots, K$:
   - Compute $\hat{S}(t+k) = \text{Model}(\mathbf{X}^{(k)})$.
   - Record $\hat{P}(\text{Attack}_{t+k})$ and $\hat{Y}_{\text{stage}, t+k}$.
   - Construct $\mathbf{X}^{(k+1)} = [\mathbf{X}^{(k)}_{2:L}, \hat{S}(t+k)]$.

Horizon-wise classification metrics ($t+1 \dots t+5$) track uncertainty propagation across forward steps.
