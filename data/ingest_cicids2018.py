import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.csv_loader import load_flow_csv
from preprocessing.window_builder import build_network_states, create_sequences
from preprocessing.scaler import StateScaler

def run_cicids2018_pipeline(raw_dir: str = "data/raw/"):
    """
    Ingestion & evaluation pipeline for real-world CIC-IDS2018 cybersecurity flow CSV files.
    """
    os.makedirs(raw_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))

    if not csv_files:
        print("=" * 80)
        print("CIC-IDS2018 REAL DATASET VALIDATION:")
        print("STATUS: NOT EXECUTED — No CSV files found in data/raw/")
        print("-" * 80)
        print("INSTRUCTIONS TO RUN REAL DATASET EVALUATION:")
        print("1. Download official CIC-IDS2018 CSV dataset files (e.g. Wednesday-14-02-2018_TrafficForML_CRC.csv).")
        print(f"2. Place the CSV file(s) into directory: {os.path.abspath(raw_dir)}")
        print("3. Execute this script: py data/ingest_cicids2018.py")
        print("=" * 80)
        return None

    print(f"Found {len(csv_files)} raw CSV file(s) in {raw_dir}: {csv_files}")
    dfs = []
    for f in csv_files:
        print(f"Processing raw flow file: {f}...")
        df_sub = load_flow_csv(f)
        dfs.append(df_sub)

    df_full = pd.concat(dfs, ignore_index=True)
    print(f"Successfully loaded real dataset with {len(df_full)} flow records.")
    return df_full

if __name__ == "__main__":
    run_cicids2018_pipeline()
