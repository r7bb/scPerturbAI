"""Pathway enrichment for differentially expressed gene sets, via Enrichr."""

from __future__ import annotations

import time

import gseapy as gp
import pandas as pd


def enrich_pathways(
    de_genes: list[str], gene_sets: str = "GO_Biological_Process_2023", max_retries: int = 5
) -> pd.DataFrame:
    """Run GO/pathway enrichment on a list of genes via the Enrichr API, retrying on rate limits."""
    if not de_genes:
        return pd.DataFrame(columns=["Term", "Overlap", "Adjusted P-value"])
    for attempt in range(max_retries):
        try:
            result = gp.enrichr(gene_list=de_genes, gene_sets=[gene_sets], organism="human", outdir=None)
            return result.results.sort_values("Adjusted P-value")
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame(columns=["Term", "Overlap", "Adjusted P-value"])


def compare_pathway_agreement(observed_pathways: pd.DataFrame, predicted_pathways: pd.DataFrame, top_n: int = 20) -> float:
    """Jaccard overlap between the top-N enriched terms of two pathway result tables."""
    obs_terms = set(observed_pathways.head(top_n)["Term"])
    pred_terms = set(predicted_pathways.head(top_n)["Term"])
    if not obs_terms and not pred_terms:
        return 0.0
    return len(obs_terms & pred_terms) / len(obs_terms | pred_terms)
