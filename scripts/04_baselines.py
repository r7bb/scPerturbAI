"""Baseline models: no-perturbation, mean-shift, and Ridge (with an embedding ablation)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.gene_embeddings import pair_embedding
from src.models.baseline import no_perturbation_baseline, mean_shift_baseline, train_ridge_baseline
from src.evaluation.metrics import evaluate_predictions

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)

deltas_npz = np.load(DATA_DIR / "pseudobulk_deltas.npz", allow_pickle=True)
hvg_genes = deltas_npz["genes"].tolist()
true_deltas = {k: deltas_npz[k] for k in deltas_npz.files if k != "genes"}
n_hvg = len(hvg_genes)

categories = {
    "train (seen, sanity check)": splits["train"],
    "test_unseen_single": splits["test_unseen_single"],
    "test_combo_0_unseen": splits["test_combo_0_unseen"],
    "test_combo_1_unseen": splits["test_combo_1_unseen"],
    "test_combo_2_unseen": splits["test_combo_2_unseen"],
}

train_perts = splits["train"]
train_matrix = np.stack([true_deltas[p] for p in train_perts])


def run_embedding_variant(embedding_path: str, label: str):
    emb = dict(np.load(MODEL_DIR / embedding_path))
    embed_dim = next(iter(emb.values())).shape[0]

    def features_for(pert: str) -> np.ndarray:
        return pair_embedding(emb, embed_dim, pert.split("_")).astype(np.float64)

    X_train = np.stack([features_for(p) for p in train_perts])
    y_train = train_matrix

    ridge = train_ridge_baseline(X_train, y_train, alpha=10.0)

    all_perts = sorted(set(p for perts in categories.values() for p in perts))
    ridge_preds = {p: ridge.predict(features_for(p)[None, :])[0] for p in all_perts}

    _, summary = evaluate_predictions(true_deltas, ridge_preds, categories)
    summary = summary.reset_index().rename(columns={"category": "category"})
    summary.insert(0, "model", f"Ridge ({label})")
    return summary


print("=== No-perturbation baseline ===")
zero_pred = {p: no_perturbation_baseline(true_deltas[p]) for perts in categories.values() for p in perts}
_, no_pert_summary = evaluate_predictions(true_deltas, zero_pred, categories)
no_pert_summary = no_pert_summary.reset_index()
no_pert_summary.insert(0, "model", "No-perturbation")

print("=== Mean-shift baseline ===")
mean_delta = mean_shift_baseline(train_matrix)
mean_pred = {p: mean_delta for perts in categories.values() for p in perts}
_, mean_summary = evaluate_predictions(true_deltas, mean_pred, categories)
mean_summary = mean_summary.reset_index()
mean_summary.insert(0, "model", "Mean-shift")

print("=== Ridge (co-expression only) ===")
ridge_coexpr_summary = run_embedding_variant("gene_embeddings_coexpr_only.npz", "co-expression only")

print("=== Ridge (co-expression + perturbation-response) ===")
ridge_full_summary = run_embedding_variant("gene_embeddings.npz", "co-expression + pert-response")

all_summaries = pd.concat(
    [no_pert_summary, mean_summary, ridge_coexpr_summary, ridge_full_summary], ignore_index=True
)
all_summaries.to_csv(TABLE_DIR / "baseline_metrics.csv", index=False)
print(all_summaries.to_string(index=False))
print("Done.")
