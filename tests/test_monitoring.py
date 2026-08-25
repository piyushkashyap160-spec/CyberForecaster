import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.drift_detector import DistributionDriftDetector

def test_drift_detector_stable():
    baseline = np.random.normal(0, 1, size=(100, 23))
    detector = DistributionDriftDetector(baseline)

    current_stable = np.random.normal(0, 1, size=(20, 23))
    res = detector.detect_drift(current_stable)
    assert 'drift_warning' in res
    assert res['drifted_features_count'] < 5

def test_drift_detector_drifted():
    baseline = np.random.normal(0, 1, size=(100, 23))
    detector = DistributionDriftDetector(baseline)

    # Shift distribution significantly for most features
    current_shifted = np.random.normal(10.0, 5.0, size=(20, 23))
    res = detector.detect_drift(current_shifted)
    assert res['drift_warning'] is True
    assert res['drifted_features_count'] > 10
