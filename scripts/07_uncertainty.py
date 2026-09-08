"""Compute ensemble mean + uncertainty (std) for all evaluated perturbations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.gene_embeddings import pair_embedding
from src.models.perturbation_model import load_model
from src.screening.uncertainty import ensemble_predict
from src.evaluation.metrics import evaluate_predictions

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

SEEDS = [11, 22, 33, 44, 55]

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)

deltas_npz = np.load(DATA_DIR / "pseudobulk_deltas.npz", allow_pickle=True)
true_deltas = {k: deltas_npz[k].astype(np.float32) for k in deltas_npz.files if k != "genes"}

print("Loading ensemble models...")
models = [load_model(MODEL_DIR / f"perturbation_model_ens{s}.pt")[0] for s in SEEDS]

emb = dict(np.load(MODEL_DIR / "gene_embeddings.npz"))
embed_dim = next(iter(emb.values())).shape[0]

hvg = sc.read_h5ad(DATA_DIR / "processed_hvg.h5ad")
control_X = np.asarray(hvg[hvg.obs["perturbation"].astype(str) == "control"].X.todense()).astype(np.float32)
rng = np.random.RandomState(0)

categories = {
    "train": splits["train"],
    "test_unseen_single": splits["test_unseen_single"],
    "test_combo_0_unseen": splits["test_combo_0_unseen"],
    "test_combo_1_unseen": splits["test_combo_1_unseen"],
    "test_combo_2_unseen": splits["test_combo_2_unseen"],
}
all_perts = sorted(set(p for perts in categories.values() for p in perts if p in true_deltas))

rows = []
mean_preds = {}
for pert in all_perts:
    pemb = pair_embedding(emb, embed_dim, pert.split("_")).astype(np.float32)
    n = min(100, control_X.shape[0])
    idx = rng.choice(control_X.shape[0], n, replace=False)
    cell = torch.tensor(control_X[idx])
    pemb_t = torch.tensor(np.tile(pemb, (n, 1)))
    mean_pred, std_pred = ensemble_predict(models, cell, pemb_t)
    mean_preds[pert] = mean_pred
    rows.append(
        {
            "perturbation": pert,
            "mean_uncertainty": float(std_pred.mean()),
            "max_predicted_effect": float(np.abs(mean_pred).max()),
        }
    )

uncertainty_df = pd.DataFrame(rows)
uncertainty_df.to_csv(TABLE_DIR / "uncertainty_summary.csv", index=False)

_, summary = evaluate_predictions(true_deltas, mean_preds, categories)
summary = summary.reset_index()
summary.insert(0, "model", "MLP ensemble (5 seeds, mean)")
summary.to_csv(TABLE_DIR / "ensemble_metrics.csv", index=False)

print(summary.to_string(index=False))
print("\nUncertainty by category (mean of per-perturbation std):")
cat_of = {p: c for c, perts in categories.items() for p in perts}
uncertainty_df["category"] = uncertainty_df["perturbation"].map(cat_of)
print(uncertainty_df.groupby("category")["mean_uncertainty"].mean())
print("Done.")
