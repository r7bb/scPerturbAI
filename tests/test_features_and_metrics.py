import numpy as np
import pytest

from src.evaluation.metrics import (
    de_gene_idx,
    de_pearson_delta,
    direction_accuracy,
    evaluate_predictions,
    mse,
    pearson_delta,
)
from src.features.gene_embeddings import pair_embedding
from src.models.baseline import mean_shift_baseline, no_perturbation_baseline, train_ridge_baseline
from src.screening.virtual_screen import rank_candidates, score_candidate


def test_pair_embedding_single_gene_pads_with_zero():
    emb = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}
    result = pair_embedding(emb, embed_dim=2, genes=["A"])
    # sum = A + 0 = A, |diff| = |A - 0| = |A|
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 1.0, 2.0]))


def test_pair_embedding_is_order_invariant():
    emb = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}
    ab = pair_embedding(emb, embed_dim=2, genes=["A", "B"])
    ba = pair_embedding(emb, embed_dim=2, genes=["B", "A"])
    np.testing.assert_allclose(ab, ba)


def test_pair_embedding_missing_gene_defaults_to_zero():
    emb = {"A": np.array([1.0, 2.0])}
    result = pair_embedding(emb, embed_dim=2, genes=["A", "UNKNOWN"])
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 1.0, 2.0]))


def test_pearson_delta_perfect_correlation():
    true_delta = np.array([1.0, 2.0, 3.0, -1.0])
    assert pearson_delta(true_delta, true_delta) == pytest.approx(1.0)


def test_pearson_delta_zero_variance_is_zero_not_nan():
    assert pearson_delta(np.zeros(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == 0.0


def test_mse_zero_for_identical_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert mse(a, a) == 0.0


def test_de_gene_idx_picks_largest_magnitude():
    true_delta = np.array([0.1, -5.0, 0.2, 3.0, 0.0])
    idx = de_gene_idx(true_delta, top_k=2)
    assert set(idx.tolist()) == {1, 3}


def test_direction_accuracy_all_correct():
    true_delta = np.array([1.0, -1.0, 2.0])
    pred_delta = np.array([0.5, -0.5, 1.0])
    idx = np.array([0, 1, 2])
    assert direction_accuracy(pred_delta, true_delta, idx) == 1.0


def test_de_pearson_delta_restricts_to_given_indices():
    true_delta = np.array([1.0, 2.0, 3.0, 4.0])
    pred_delta = np.array([1.0, 2.0, -3.0, -4.0])
    idx = np.array([0, 1])
    assert de_pearson_delta(pred_delta, true_delta, idx) == pytest.approx(1.0)


def test_evaluate_predictions_aggregates_by_category():
    true_deltas = {"p1": np.array([1.0, 2.0, 3.0]), "p2": np.array([1.0, -2.0, 3.0])}
    pred_deltas = {"p1": np.array([1.0, 2.0, 3.0]), "p2": np.array([1.0, -2.0, 3.0])}
    categories = {"catA": ["p1"], "catB": ["p2"]}
    per_pert, summary = evaluate_predictions(true_deltas, pred_deltas, categories, top_k=2)
    assert len(per_pert) == 2
    assert set(summary.index) == {"catA", "catB"}
    assert (summary["pearson_delta"] > 0.99).all()


def test_no_perturbation_baseline_is_zero():
    control = np.array([1.0, 2.0, 3.0])
    result = no_perturbation_baseline(control)
    np.testing.assert_array_equal(result, np.zeros(3))


def test_mean_shift_baseline_averages_rows():
    deltas = np.array([[1.0, 1.0], [3.0, 3.0]])
    np.testing.assert_allclose(mean_shift_baseline(deltas), np.array([2.0, 2.0]))


def test_ridge_baseline_fits_linear_relationship():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    true_w = rng.randn(4, 3)
    y = X @ true_w
    model = train_ridge_baseline(X, y, alpha=0.01)
    preds = model.predict(X)
    assert pearson_delta(preds.ravel(), y.ravel()) > 0.99


def test_score_candidate_increases_with_effect_and_relevance():
    low = score_candidate(effect=0.1, pathway_relevance=0.1, uncertainty=0.5)
    high = score_candidate(effect=0.9, pathway_relevance=0.9, uncertainty=0.5)
    assert high > low


def test_score_candidate_decreases_with_uncertainty():
    confident = score_candidate(effect=0.5, pathway_relevance=0.5, uncertainty=0.0)
    unsure = score_candidate(effect=0.5, pathway_relevance=0.5, uncertainty=5.0)
    assert confident > unsure


def test_rank_candidates_sorts_descending_by_priority():
    import pandas as pd

    df = pd.DataFrame(
        {
            "candidate": ["low", "high"],
            "effect": [0.1, 0.9],
            "pathway_relevance": [0.1, 0.9],
            "uncertainty": [0.5, 0.1],
        }
    )
    ranked = rank_candidates(df)
    assert ranked.iloc[0]["candidate"] == "high"
    assert ranked["priority"].is_monotonic_decreasing

