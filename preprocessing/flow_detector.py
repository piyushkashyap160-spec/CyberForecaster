"""
flow_detector.py
================
Fast per-flow anomaly/suspicion detector interface for CyberForecaster.

Evaluates individual network flows immediately upon capture (e.g. via XGBoost or Random Forest),
complementing the Temporal LSTM World Model which predicts multi-step future state trajectories.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("cyberforecaster.flow_detector")

class FastFlowDetector:
    """
    Per-flow suspicion evaluator.
    If a trained model checkpoint exists at `models_weights/flow_detector.joblib` or `.json`,
    it evaluates individual flow features. Otherwise, reports model unavailable without fabrication.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "..", "models_weights", "flow_detector.joblib"
        )
        self.model = None
        self.is_configured = False
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                import joblib
                payload = joblib.load(self.model_path)
                if isinstance(payload, dict) and "model" in payload:
                    self.model = payload["model"]
                    self.metadata = payload
                else:
                    self.model = payload
                    self.metadata = {}
                self.is_configured = True
                logger.info(f"FastFlowDetector loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load flow detector from {self.model_path}: {e}")
                self.is_configured = False
                self.model = None
        else:
            logger.info(f"FastFlowDetector checkpoint not present at {self.model_path}. Fast detector marked NOT_CONFIGURED.")
            self.is_configured = False
            self.model = None

    def extract_features(self, flow: Dict[str, Any]) -> list:
        """
        Extracts the 8-D live-compatible feature vector in exact training order:
        1. duration_sec
        2. byte_count
        3. packet_count
        4. is_tcp
        5. avg_packet_size
        6. syn_count
        7. ack_count
        8. rst_count
        """
        dur = max(0.0001, float(flow.get("duration", 0.001)))
        bytes_cnt = float(flow.get("byte_count", 0.0))
        pkts_cnt = float(flow.get("packet_count", 1.0))
        is_tcp = 1.0 if str(flow.get("protocol", "")).upper() == "TCP" else 0.0
        avg_pkt = float(flow.get("avg_packet_size", bytes_cnt / max(1.0, pkts_cnt)))
        syn_cnt = float(flow.get("syn_count", 0.0))
        ack_cnt = float(flow.get("ack_count", 0.0))
        rst_cnt = float(flow.get("rst_count", 0.0))

        return [
            dur,
            bytes_cnt,
            pkts_cnt,
            is_tcp,
            avg_pkt,
            syn_cnt,
            ack_cnt,
            rst_cnt
        ]

    def predict_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates an individual flow using the 8-D feature vector.
        Returns evaluation metadata or explicit NOT_CONFIGURED status.
        """
        if not self.is_configured or self.model is None:
            return {
                "available": False,
                "status": "NOT_CONFIGURED",
                "suspicious": False,
                "confidence": 0.0,
                "reason": "Fast per-flow detector model checkpoint not configured"
            }

        try:
            features = self.extract_features(flow)
            pred = int(self.model.predict([features])[0])
            prob = float(self.model.predict_proba([features])[0][1]) if hasattr(self.model, "predict_proba") else (1.0 if pred == 1 else 0.0)
            return {
                "available": True,
                "status": "ACTIVE",
                "suspicious": bool(pred == 1 or prob >= 0.5),
                "confidence": round(prob, 4),
                "predicted_label": "Bot / Suspicious" if (pred == 1 or prob >= 0.5) else "Benign",
                "reason": "Flow evaluated by fast per-flow model"
            }
        except Exception as e:
            logger.error(f"Error during fast flow prediction: {e}")
            return {
                "available": False,
                "status": "ERROR",
                "suspicious": False,
                "confidence": 0.0,
                "reason": f"Evaluation error: {e}"
            }

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_type": "Fast Per-Flow Detector (RandomForest)",
            "is_configured": self.is_configured,
            "checkpoint_path": self.model_path,
            "status": "ACTIVE" if self.is_configured else "NOT_CONFIGURED"
        }
