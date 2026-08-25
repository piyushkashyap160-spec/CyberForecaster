import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.scaler import StateScaler
from models.baseline_model import StaticLogisticRegressionBaseline, TemporalLogisticRegressionBaseline

def train_baseline(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    csv_path = config['data']['demo_csv_path']
    df = load_flow_csv(csv_path)

    window_sec = config['data']['window_seconds']
    seq_len = config['sequence']['sequence_length']

    states = build_network_states(df, window_seconds=window_sec)
    X, y_state, y_attack, y_stage, _ = create_sequences(states, sequence_length=seq_len)

    total_samples = len(X)
    train_end = int(total_samples * config['training']['train_split'])
    X_train, y_attack_train = X[:train_end], y_attack[:train_end]

    scaler = StateScaler()
    scaler.load(config['model']['scaler_path'])
    X_train_scaled = scaler.transform(X_train)

    # 1. Baseline A: Static Logistic Regression S(t)
    static_baseline = StaticLogisticRegressionBaseline()
    static_baseline.fit(X_train_scaled, y_attack_train)
    static_path = config['model']['baseline_path']
    static_baseline.save(static_path)
    print(f"Static LR Baseline (Baseline A) trained and saved to {static_path}")

    # 2. Baseline B: Temporal Logistic Regression [S(t-9)...S(t)]
    temporal_baseline = TemporalLogisticRegressionBaseline()
    temporal_baseline.fit(X_train_scaled, y_attack_train)
    temporal_path = "models_weights/baseline_temporal_lr.joblib"
    temporal_baseline.save(temporal_path)
    print(f"Temporal LR Baseline (Baseline B) trained and saved to {temporal_path}")

if __name__ == "__main__":
    train_baseline()
