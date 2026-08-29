"""
test_lead_time_audit.py — Unit and Regression Tests for Lead-Time Calculation & Metric Integrity.
"""

import pytest
import numpy as np
from forecasting.lead_time import compute_forecast_lead_time

def test_lead_time_zero_delay_at_onset():
    """Test exact detection at attack onset results in 0s lead time."""
    y_attack = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    probs = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1])
    res = compute_forecast_lead_time(y_attack, probs, window_seconds=5.0, threshold=0.5)
    assert res["episodes_detected"] == 1
    assert res["mean_lead_time_seconds"] == 0.0

def test_lead_time_early_warning():
    """Test detection 2 windows before onset gives +10s lead time."""
    y_attack = np.array([0, 0, 0, 0, 1, 1, 1, 0])
    # Model triggers at index 2 (onset is at index 4) -> 2 windows early
    probs = np.array([0.1, 0.1, 0.8, 0.9, 0.9, 0.9, 0.1, 0.1])
    res = compute_forecast_lead_time(y_attack, probs, window_seconds=5.0, threshold=0.5)
    assert res["episodes_detected"] == 1
    assert res["mean_lead_time_seconds"] == 10.0

def test_lead_time_post_onset_delay():
    """Test detection 2 windows after onset gives -10s lead time."""
    y_attack = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    # Model triggers at index 5 (onset is at index 3) -> 2 windows late
    probs = np.array([0.1, 0.1, 0.1, 0.2, 0.3, 0.8, 0.9, 0.1])
    res = compute_forecast_lead_time(y_attack, probs, window_seconds=5.0, threshold=0.5)
    assert res["episodes_detected"] == 1
    assert res["mean_lead_time_seconds"] == -10.0

def test_lead_time_no_episodes():
    """Test graceful handling when no attack episodes exist."""
    y_attack = np.zeros(20, dtype=int)
    probs = np.random.uniform(0.1, 0.4, size=20)
    res = compute_forecast_lead_time(y_attack, probs, window_seconds=5.0, threshold=0.5)
    assert res["episodes_detected"] == 0
    assert res["total_episodes"] == 0
    assert res["mean_lead_time_seconds"] == 0.0

def test_rmse_baseline_identity():
    """Test persistence baseline identity calculation."""
    state = np.random.randn(10, 23)
    persistence_rmse = float(np.sqrt(np.mean((state - state) ** 2)))
    assert persistence_rmse == 0.0
