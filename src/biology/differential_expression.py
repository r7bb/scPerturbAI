"""Differential expression and E-distance between perturbed and control cells."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist


def run_de(adata, perturbation: str, control_label: str = "control", perturbation_col: str = "perturbation") -> pd.DataFrame:
    """Rank genes differentially expressed in `perturbation` vs. control (Wilcoxon test)."""
    subset = adata[adata.obs[perturbation_col].isin([perturbation, control_label])].copy()
    subset.obs[perturbation_col] = subset.obs[perturbation_col].astype(str)
    sc.tl.rank_genes_groups(
        subset, groupby=perturbation_col, groups=[perturbation], reference=control_label, method="wilcoxon"
    )
    result = subset.uns["rank_genes_groups"]
    return pd.DataFrame(
        {
            "gene": result["names"][perturbation],
            "score": result["scores"][perturbation],
            "logfoldchange": result["logfoldchanges"][perturbation],
            "pval_adj": result["pvals_adj"][perturbation],
        }
    )


def e_distance(
    embedding: np.ndarray,
    group_a_idx: np.ndarray,
    group_b_idx: np.ndarray,
    max_cells: int = 300,
    seed: int = 0,
) -> float:
    """Energy-distance statistic between two groups of cells in a shared embedding space.

    E(A, B) = 2 * mean(d(A, B)) - mean(d(A, A)) - mean(d(B, B))
    """
    rng = np.random.RandomState(seed)
    if len(group_a_idx) > max_cells:
        group_a_idx = rng.choice(group_a_idx, max_cells, replace=False)
    if len(group_b_idx) > max_cells:
        group_b_idx = rng.choice(group_b_idx, max_cells, replace=False)

    a = embedding[group_a_idx]
    b = embedding[group_b_idx]

    d_ab = cdist(a, b).mean()
    d_aa = cdist(a, a).mean()
    d_bb = cdist(b, b).mean()
    return float(2 * d_ab - d_aa - d_bb)
