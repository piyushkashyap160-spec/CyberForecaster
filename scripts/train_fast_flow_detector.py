"""
train_fast_flow_detector.py
===========================
Trains an auxiliary per-flow anomaly/threat detector (RandomForest) on genuine CIC-IDS2018 flow records.

Features (8-D Live-Compatible Vector):
1. duration_sec       (Flow Duration in seconds: microsec / 1e6)
2. byte_count         (TotLen Fwd Pkts + TotLen Bwd Pkts)
3. packet_count       (Tot Fwd Pkts + Tot Bwd Pkts)
4. is_tcp             (1.0 if Protocol == 6 else 0.0)
5. avg_packet_size    (Pkt Size Avg)
6. syn_count          (SYN Flag Cnt)
7. ack_count          (ACK Flag Cnt)
8. rst_count          (RST Flag Cnt)

Ground Truth Labels:
0 = Benign
1 = Bot (from Friday-02-03-2018_TrafficForML_CICFlowMeter.csv)
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_flow_detector")

FEATURE_COLUMNS = [
    "duration_sec",
    "byte_count",
    "packet_count",
    "is_tcp",
    "avg_packet_size",
    "syn_count",
    "ack_count",
    "rst_count"
]

def load_and_preprocess_dataset(csv_path: str):
    logger.info(f"Loading raw dataset from {csv_path}...")
    cols_to_load = [
        "Flow Duration",
        "Tot Fwd Pkts", "Tot Bwd Pkts",
        "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Protocol",
        "Pkt Size Avg",
        "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt",
        "Label"
    ]
    df_raw = pd.read_csv(csv_path, usecols=cols_to_load)
    raw_total = len(df_raw)
    logger.info(f"Raw rows loaded: {raw_total}")

    # Handle missing or invalid labels
    df_raw = df_raw.dropna(subset=["Label"]).copy()

    # Feature construction matching live FlowAggregator semantics
    df = pd.DataFrame()
    df["duration_sec"] = (df_raw["Flow Duration"] / 1e6).clip(lower=0.0001)
    df["byte_count"] = (df_raw["TotLen Fwd Pkts"] + df_raw["TotLen Bwd Pkts"]).astype(np.float32)
    df["packet_count"] = (df_raw["Tot Fwd Pkts"] + df_raw["Tot Bwd Pkts"]).astype(np.float32)
    df["is_tcp"] = (df_raw["Protocol"] == 6).astype(np.float32)
    df["avg_packet_size"] = df_raw["Pkt Size Avg"].astype(np.float32)
    df["syn_count"] = df_raw["SYN Flag Cnt"].astype(np.float32)
    df["ack_count"] = df_raw["ACK Flag Cnt"].astype(np.float32)
    df["rst_count"] = df_raw["RST Flag Cnt"].astype(np.float32)

    # Binary target: Benign=0, Bot=1
    df["target"] = (df_raw["Label"] == "Bot").astype(int)

    # Drop NaNs or Infs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info("Class distribution BEFORE deduplication:")
    counts_before = df["target"].value_counts().to_dict()
    logger.info(f"  Benign (0): {counts_before.get(0, 0)} ({counts_before.get(0, 0)/len(df)*100:.2f}%)")
    logger.info(f"  Bot (1):    {counts_before.get(1, 0)} ({counts_before.get(1, 0)/len(df)*100:.2f}%)")

    # Deduplication across the 8 feature columns
    before_dedup = len(df)
    df_dedup = df.drop_duplicates(subset=FEATURE_COLUMNS).copy()
    after_dedup = len(df_dedup)
    duplicates_removed = before_dedup - after_dedup
    logger.info(f"Deduplication: removed {duplicates_removed} duplicate feature rows ({duplicates_removed/before_dedup*100:.2f}%).")
    logger.info(f"Rows remaining after deduplication: {after_dedup}")

    counts_after = df_dedup["target"].value_counts().to_dict()
    logger.info("Class distribution AFTER deduplication:")
    logger.info(f"  Benign (0): {counts_after.get(0, 0)} ({counts_after.get(0, 0)/after_dedup*100:.2f}%)")
    logger.info(f"  Bot (1):    {counts_after.get(1, 0)} ({counts_after.get(1, 0)/after_dedup*100:.2f}%)")

    return df_dedup, {
        "raw_rows": raw_total,
        "before_dedup_rows": before_dedup,
        "after_dedup_rows": after_dedup,
        "duplicates_removed": duplicates_removed,
        "class_counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "class_counts_after": {str(k): int(v) for k, v in counts_after.items()}
    }

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_root, "data", "real", "cicids2018", "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
    out_model_path = os.path.join(repo_root, "models_weights", "flow_detector.joblib")
    out_metrics_path = os.path.join(repo_root, "experiments", "results", "flow_detector_evaluation.json")

    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        sys.exit(1)

    df_clean, data_stats = load_and_preprocess_dataset(csv_path)

    X = df_clean[FEATURE_COLUMNS].values
    y = df_clean["target"].values

    # Stratified Train-Test Split (80/20)
    logger.info("Performing Stratified 80/20 Train-Test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    logger.info(f"Training samples: {len(X_train)} (Benign={np.sum(y_train==0)}, Bot={np.sum(y_train==1)})")
    logger.info(f"Testing samples:  {len(X_test)} (Benign={np.sum(y_test==0)}, Bot={np.sum(y_test==1)})")

    # Train compact balanced Random Forest Classifier
    logger.info("Training balanced RandomForestClassifier (n_estimators=100, max_depth=12)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate on held-out test set
    logger.info("Evaluating on held-out test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    logger.info(f"Test Precision: {prec:.4f}")
    logger.info(f"Test Recall:    {rec:.4f}")
    logger.info(f"Test F1-Score:  {f1:.4f}")
    logger.info(f"Test ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"False Positive Rate (FPR): {fpr:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Package model payload
    model_payload = {
        "model": clf,
        "feature_names": FEATURE_COLUMNS,
        "model_type": "RandomForestClassifier",
        "n_features": len(FEATURE_COLUMNS),
        "target_mapping": {"0": "Benign", "1": "Bot"},
        "trained_on": "CIC-IDS2018 (Friday-02-03-2018)",
        "training_date": "2026-08-30"
    }

    # Save artifact
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    joblib.dump(model_payload, out_model_path)
    logger.info(f"Flow detector model saved to {out_model_path}")

    # Save evaluation report
    eval_results = {
        "dataset_stats": data_stats,
        "features": FEATURE_COLUMNS,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "metrics": {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "false_positive_rate": fpr,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        },
        "model_parameters": {
            "n_estimators": 100,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": 42
        }
    }
    os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
    with open(out_metrics_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluation report saved to {out_metrics_path}")

if __name__ == "__main__":
    main()
