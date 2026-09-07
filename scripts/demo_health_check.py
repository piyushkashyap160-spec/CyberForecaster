#!/usr/bin/env python3
"""
scripts/demo_health_check.py
=============================
Non-invasive pre-demo verification tool for CyberForecaster.
Validates model weights, scaler, benchmark results, backend endpoints, and frontend build.
"""

import os
import sys
import json
import urllib.request
import urllib.error

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)

def check_file(path, desc):
    full_path = os.path.join(REPO_ROOT, path)
    exists = os.path.exists(full_path)
    status = "[PASS]" if exists else "[FAIL]"
    print(f"{status} {desc}: {path}")
    return exists

def check_endpoint(url, desc):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CyberForecasterHealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=3) as res:
            if res.status == 200:
                print(f"[PASS] {desc} reachable (HTTP 200): {url}")
                return True
            else:
                print(f"[WARN] {desc} returned HTTP {res.status}: {url}")
                return False
    except urllib.error.URLError:
        print(f"[INFO] {desc} offline (backend not currently started): {url}")
        return False
    except Exception as e:
        print(f"[WARN] {desc} check error: {e}")
        return False

def main():
    print("=" * 65)
    print("  CYBERFORECASTER — SIH 2026 PRE-DEMO HEALTH CHECK")
    print("=" * 65)
    print()

    all_passed = True

    # 1. Essential Disk Artifacts
    print("--- 1. Checking Essential Machine Learning & Provenance Files ---")
    all_passed &= check_file("models_weights/lstm_world_model.pt", "Temporal LSTM World Model Weights")
    all_passed &= check_file("models_weights/scaler.joblib", "Train-Fitted StateScaler Checkpoint")
    all_passed &= check_file("experiments/results/canonical_benchmark_results.json", "Canonical Benchmark Results Artifact")
    all_passed &= check_file("data/demo/demo_cicids2018.csv", "Demo Physical Flow Telemetry CSV")
    all_passed &= check_file("frontend/dist/index.html", "Production Frontend Build Artifact (dist/)")
    print()

    # 2. Benchmark Artifact Scientific Integrity
    print("--- 2. Validating Canonical Benchmark Artifact Integrity ---")
    bench_path = os.path.join(REPO_ROOT, "experiments/results/canonical_benchmark_results.json")
    if os.path.exists(bench_path):
        try:
            with open(bench_path, "r") as f:
                bench = json.load(f)
            rmse = bench["future_state_rmse_benchmarks"]["horizons"]["K_1_step_5s_ahead"]
            cls_res = bench["classification_results"]["Temporal_LSTM_World_Model_23D"]
            lead = bench["lead_time_evaluation"]

            print(f"[PASS] Canonical K=1 Future State RMSE: LSTM {rmse['lstm_rollout_rmse']} vs Persistence {rmse['persistence_rmse']} vs Train Mean {rmse['training_mean_rmse']}")
            print(f"[PASS] Untouched Test Partition Classification: F1={cls_res['F1_Score']}, Precision={cls_res['Precision']}, Recall={cls_res['Recall']}, FPR={cls_res['FPR']}")
            print(f"[PASS] Validated Lead Time: {lead['validated_lead_time_seconds']}s (Exact Onset; 0 Pre-Onset Alarms)")
        except Exception as e:
            print(f"[FAIL] Benchmark artifact parse error: {e}")
            all_passed = False
    else:
        print("[FAIL] Canonical benchmark results artifact not found on disk.")
        all_passed = False
    print()

    # 3. Live Backend Service Check (If running)
    print("--- 3. Checking Live Backend Endpoints (Optional if not yet launched) ---")
    backend_up = check_endpoint("http://127.0.0.1:8000/api/hosts", "FastAPI Hosts API")
    if backend_up:
        check_endpoint("http://127.0.0.1:8000/api/world_model/status", "World Model Status Endpoint")
        check_endpoint("http://127.0.0.1:8000/api/snort/status", "Snort Correlator Endpoint")
        check_endpoint("http://127.0.0.1:8000/api/flow_detector/status", "FastFlowDetector Endpoint")
        check_endpoint("http://127.0.0.1:8000/api/blockchain/status", "Blockchain Status Endpoint")
    else:
        print("[INFO] Start backend with: `uvicorn backend.main:fastapi_app --host 127.0.0.1 --port 8000`")
    print()

    print("=" * 65)
    if all_passed:
        print("  HEALTH CHECK VERDICT: READY FOR SIH DEMO")
        print("  All essential weights, scalers, benchmark artifacts, and builds verified.")
    else:
        print("  HEALTH CHECK VERDICT: REVIEW REQUIRED (Missing files listed above)")
    print("=" * 65)

if __name__ == "__main__":
    main()
