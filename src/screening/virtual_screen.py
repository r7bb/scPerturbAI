"""Rank candidate (untested) gene perturbations for follow-up experiments."""

from __future__ import annotations

import pandas as pd


def score_candidate(effect: float, pathway_relevance: float, uncertainty: float) -> float:
    """Priority score combining predicted effect size, pathway relevance, and (inverse) uncertainty."""
    confidence = 1.0 / (1.0 + uncertainty)
    return effect * pathway_relevance * confidence


def rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Score and rank a table of candidate perturbations, expects effect/pathway_relevance/uncertainty columns."""
    candidates = candidates.copy()
    candidates["priority"] = candidates.apply(
        lambda row: score_candidate(row["effect"], row["pathway_relevance"], row["uncertainty"]),
        axis=1,
    )
    return candidates.sort_values("priority", ascending=False)
