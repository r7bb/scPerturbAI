"""Ensemble-based uncertainty estimation for perturbation predictions."""

from __future__ import annotations

import numpy as np
import torch


def ensemble_predict(
    models: list[torch.nn.Module], control_cells: torch.Tensor, pert_embedding: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Run all ensemble members on the same (control_cells, pert_embedding) batch.

    Returns (mean_prediction, std_uncertainty), each averaged over the cell batch too.
    """
    predictions = []
    with torch.no_grad():
        for model in models:
            model.eval()
            pred = model(control_cells, pert_embedding).numpy()
            predictions.append(pred.mean(axis=0))
    predictions = np.stack(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0)
