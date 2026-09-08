"""Loading, QC, and normalization for the Norman Perturb-seq AnnData object."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc


def load_dataset(path: str) -> ad.AnnData:
    """Load the harmonized scPerturb .h5ad file."""
    return sc.read_h5ad(path)


def run_qc(adata: ad.AnnData, min_genes: int = 500, max_percent_mito: float = 20.0) -> ad.AnnData:
    """Drop low-quality cells using the QC metrics already computed by scPerturb."""
    keep = (adata.obs["ngenes"] >= min_genes) & (adata.obs["percent_mito"] <= max_percent_mito)
    return adata[keep].copy()


def normalize(adata: ad.AnnData, n_top_genes: int = 2000) -> tuple[ad.AnnData, ad.AnnData]:
    """Library-size normalize + log1p on all genes, then return (full, hvg_only) views.

    The full log-normalized matrix is kept for differential-expression / pathway analysis
    (which is more informative over all genes); the HVG-only matrix is used for modeling.
    """
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    hvg = adata[:, adata.var["highly_variable"]].copy()
    return adata, hvg


def perturbation_genes(perturbation: str) -> list[str]:
    """Split a perturbation label ('control', 'GENE', or 'GENE1_GENE2') into constituent genes."""
    if perturbation == "control":
        return []
    return perturbation.split("_")


def pseudobulk_deltas(hvg: ad.AnnData, perturbation_col: str = "perturbation") -> dict[str, np.ndarray]:
    """Mean HVG expression per perturbation minus mean control expression."""
    control_mean = np.asarray(hvg[hvg.obs[perturbation_col] == "control"].X.mean(axis=0)).ravel()
    deltas = {}
    for pert in hvg.obs[perturbation_col].unique():
        if pert == "control":
            continue
        mean_expr = np.asarray(hvg[hvg.obs[perturbation_col] == pert].X.mean(axis=0)).ravel()
        deltas[pert] = mean_expr - control_mean
    return deltas
