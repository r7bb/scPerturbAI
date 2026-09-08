"""Simple baselines: no-op, mean-shift, and Ridge regression on perturbation features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def no_perturbation_baseline(control_expression: np.ndarray) -> np.ndarray:
    """Predict zero change (control expression unchanged)."""
    return np.zeros_like(control_expression)


def mean_shift_baseline(train_deltas: np.ndarray) -> np.ndarray:
    """Predict the average observed delta across all training perturbations."""
    return train_deltas.mean(axis=0)


def train_ridge_baseline(features: np.ndarray, deltas: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Fit a Ridge model mapping perturbation features to expression deltas."""
    model = Ridge(alpha=alpha)
    model.fit(features, deltas)
    return model
