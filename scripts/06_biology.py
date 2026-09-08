"""Differential expression + pathway enrichment: observed vs. model-predicted DE genes."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.biology.differential_expression import run_de
from src.biology.pathway_analysis import enrich_pathways, compare_pathway_agreement
from src.features.gene_embeddings import pair_embedding
from src.models.perturbation_model import load_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)

e_dist = pd.read_csv(TABLE_DIR / "e_distance.csv").set_index("perturbation")["e_distance"]

deltas_npz = np.load(DATA_DIR / "pseudobulk_deltas.npz", allow_pickle=True)
hvg_genes = deltas_npz["genes"].tolist()
true_deltas = {k: deltas_npz[k].astype(np.float32) for k in deltas_npz.files if k != "genes"}

model, ckpt = load_model(MODEL_DIR / "perturbation_model_full.pt")
emb = dict(np.load(MODEL_DIR / "gene_embeddings.npz"))
embed_dim = next(iter(emb.values())).shape[0]

hvg = sc.read_h5ad(DATA_DIR / "processed_hvg.h5ad")
control_X = np.asarray(hvg[hvg.obs["perturbation"].astype(str) == "control"].X.todense()).astype(np.float32)
rng = np.random.RandomState(0)


def predict_delta(pert: str) -> np.ndarray:
    pemb = pair_embedding(emb, embed_dim, pert.split("_")).astype(np.float32)
    n = min(100, control_X.shape[0])
    idx = rng.choice(control_X.shape[0], n, replace=False)
    with torch.no_grad():
        cell = torch.tensor(control_X[idx])
        pemb_t = torch.tensor(np.tile(pemb, (n, 1)))
        pred = model(cell, pemb_t).numpy()
    return pred.mean(axis=0)


category_of = {}
for cat in ["train", "test_unseen_single", "test_combo_0_unseen", "test_combo_1_unseen", "test_combo_2_unseen"]:
    for p in splits[cat]:
        category_of[p] = cat

candidates = [p for p in e_dist.index if p in true_deltas]
top_by_effect = e_dist.loc[candidates].sort_values(ascending=False).head(3).index.tolist()
one_per_category = []
for cat in ["test_combo_0_unseen", "test_combo_1_unseen", "test_unseen_single"]:
    perts_in_cat = [p for p in splits[cat] if p in e_dist.index]
    if perts_in_cat:
        one_per_category.append(max(perts_in_cat, key=lambda p: e_dist.get(p, 0)))

example_perturbations = list(dict.fromkeys(top_by_effect + one_per_category))
print(f"Analyzing {len(example_perturbations)} example perturbations: {example_perturbations}")

print("Loading full log-normalized matrix for DE testing...")
full = sc.read_h5ad(DATA_DIR / "processed_full.h5ad")

rows = []
for pert in example_perturbations:
    print(f"\n=== {pert} (category: {category_of.get(pert, 'train')}) ===")
    de_df = run_de(full, pert)
    observed_genes = de_df.reindex(de_df["score"].abs().sort_values(ascending=False).index).head(50)["gene"].tolist()

    pred_delta = predict_delta(pert)
    top_idx = np.argsort(-np.abs(pred_delta))[:50]
    predicted_genes = [hvg_genes[i] for i in top_idx]

    observed_pathways = enrich_pathways(observed_genes)
    time.sleep(3)
    predicted_pathways = enrich_pathways(predicted_genes)
    time.sleep(3)
    agreement = compare_pathway_agreement(observed_pathways, predicted_pathways, top_n=20)

    print("Top observed pathways:", observed_pathways["Term"].head(3).tolist())
    print("Top predicted pathways:", predicted_pathways["Term"].head(3).tolist())
    print(f"Top-20 pathway Jaccard overlap: {agreement:.3f}")

    rows.append(
        {
            "perturbation": pert,
            "category": category_of.get(pert, "train"),
            "e_distance": e_dist.get(pert, np.nan),
            "top_observed_pathway": observed_pathways["Term"].iloc[0] if len(observed_pathways) else "",
            "top_predicted_pathway": predicted_pathways["Term"].iloc[0] if len(predicted_pathways) else "",
            "pathway_jaccard_top20": agreement,
        }
    )

result_df = pd.DataFrame(rows)
result_df.to_csv(TABLE_DIR / "pathway_agreement.csv", index=False)
print("\n", result_df.to_string(index=False))
print("Done.")
