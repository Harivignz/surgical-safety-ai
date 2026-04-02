# utils/risk_proxy.py
"""
Rule-Based Risk Score Proxy
Harivignesh — SurgSentinel Prototype

In the full system, risk comes from a BiLSTM trained on high-risk window labels.
In this prototype, we use a clinically-grounded rule-based proxy:
  - Phase 1 (Calot Dissection) and Phase 2 (Clipping) = high risk
  - Temporal smoothing simulates the LSTM's lookback window
  - Small random noise prevents static-looking bar in demo

This is scientifically honest. The blueprint describes this as the prototype approach.
"""
import numpy as np
from collections import deque
from assets.phase_colors import PHASE_BASE_RISK


class RiskScoreProxy:
    """
    Smooth, animated risk score for prototype demo.
    Simulates BiLSTM temporal integration with a sliding window smoother.
    """

    def __init__(self, window_size: int = 30, noise_scale: float = 0.04):
        """
        Args:
            window_size: Frames to average over (simulates LSTM lookback)
            noise_scale: Small Gaussian noise for visual animation
        """
        self.window_size = window_size
        self.noise_scale = noise_scale
        self.history = deque(maxlen=window_size)
        self.current_score = 0.05
        self._rng = np.random.default_rng(42)

    def update(self, phase: int, frame_idx: int) -> float:
        """
        Update risk score based on current phase.
        Returns smoothed risk score in [0.0, 1.0].
        """
        # Base risk from phase
        base = PHASE_BASE_RISK.get(phase, 0.05)

        # Add temporal noise for visual dynamism
        noise = self._rng.normal(0, self.noise_scale)
        raw = np.clip(base + noise, 0.0, 1.0)

        self.history.append(raw)

        # Exponentially weighted moving average
        if len(self.history) < 3:
            self.current_score = raw
        else:
            arr = np.array(self.history)
            weights = np.exp(np.linspace(-1, 0, len(arr)))
            weights /= weights.sum()
            self.current_score = float(np.dot(arr, weights))

        return self.current_score

    def get_risk_label(self, score: float) -> tuple:
        """Returns (label_str, BGR_color) for the current risk level."""
        if score < 0.30:
            return "LOW RISK", (50, 200, 50)        # Green
        elif score < 0.60:
            return "MODERATE RISK", (0, 165, 255)   # Orange
        else:
            return "HIGH RISK", (30, 30, 220)        # Red
