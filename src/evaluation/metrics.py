"""Evaluation metrics for predicted vs. observed expression deltas."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def pearson_delta(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Pearson correlation between predicted and observed expression change."""
    if np.std(pred_delta) < 1e-12 or np.std(true_delta) < 1e-12:
        return 0.0
    return float(pearsonr(pred_delta.ravel(), true_delta.ravel())[0])


def mse(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Mean squared error between predicted and observed expression change."""
    return float(np.mean((pred_delta - true_delta) ** 2))


def de_gene_idx(true_delta: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Indices of the top-k genes by absolute true delta (proxy for DE genes)."""
    return np.argsort(-np.abs(true_delta))[:top_k]


def de_pearson_delta(pred_delta: np.ndarray, true_delta: np.ndarray, gene_idx: np.ndarray) -> float:
    """Pearson correlation restricted to the most differentially expressed genes."""
    return pearson_delta(pred_delta[gene_idx], true_delta[gene_idx])


def direction_accuracy(pred_delta: np.ndarray, true_delta: np.ndarray, gene_idx: np.ndarray) -> float:
    """Fraction of DE genes where predicted and observed changes share sign."""
    pred_sign = np.sign(pred_delta[gene_idx])
    true_sign = np.sign(true_delta[gene_idx])
    return float(np.mean(pred_sign == true_sign))


def evaluate_predictions(
    true_deltas: dict[str, np.ndarray],
    pred_deltas: dict[str, np.ndarray],
    categories: dict[str, list[str]],
    top_k: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-perturbation metrics, aggregated to a per-category mean table.

    Returns (per_perturbation_df, per_category_summary_df).
    """
    rows = []
    for category, perts in categories.items():
        for pert in perts:
            if pert not in true_deltas or pert not in pred_deltas:
                continue
            true_d, pred_d = true_deltas[pert], pred_deltas[pert]
            idx = de_gene_idx(true_d, top_k=top_k)
            rows.append(
                {
                    "category": category,
                    "perturbation": pert,
                    "pearson_delta": pearson_delta(pred_d, true_d),
                    "mse": mse(pred_d, true_d),
                    "de_pearson_delta": de_pearson_delta(pred_d, true_d, idx),
                    "direction_accuracy": direction_accuracy(pred_d, true_d, idx),
                }
            )
    per_pert = pd.DataFrame(rows)
    if per_pert.empty:
        return per_pert, per_pert
    summary = (
        per_pert.groupby("category")[["pearson_delta", "mse", "de_pearson_delta", "direction_accuracy"]]
        .mean()
        .assign(n_perturbations=per_pert.groupby("category").size())
    )
    return per_pert, summary
