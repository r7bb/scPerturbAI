"""Generate and rank untested gene-pair candidates for follow-up CRISPR experiments."""
import itertools
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
from src.screening.virtual_screen import rank_candidates

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

SEEDS = [11, 22, 33, 44, 55]

# Canonical erythroid-differentiation marker genes (K562 is an erythroleukemia line),
# used as an example "target pathway" for prioritization. Swappable for any gene list.
ERYTHROID_MARKERS = {
    "GATA1", "KLF1", "TAL1", "GATA2", "EPOR", "HBB", "HBA1", "HBA2",
    "ALAS2", "SPTA1", "SLC4A1", "BCL11A", "NFE2", "TFRC", "GYPA", "EPB42",
}

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)
existing_perturbations = set(splits["train"] + splits["test_unseen_single"] + splits["test_combo_0_unseen"] + splits["test_combo_1_unseen"] + splits["test_combo_2_unseen"])

deltas_npz = np.load(DATA_DIR / "pseudobulk_deltas.npz", allow_pickle=True)
hvg_genes = deltas_npz["genes"].tolist()
found_markers = [g for g in ERYTHROID_MARKERS if g in hvg_genes]
print(f"Erythroid marker genes present in HVG set: {len(found_markers)} / {len(ERYTHROID_MARKERS)}")

emb = dict(np.load(MODEL_DIR / "gene_embeddings.npz"))
embed_dim = next(iter(emb.values())).shape[0]
vocab_genes = sorted(emb.keys())

print("Loading ensemble models...")
models = [load_model(MODEL_DIR / f"perturbation_model_ens{s}.pt")[0] for s in SEEDS]

hvg = sc.read_h5ad(DATA_DIR / "processed_hvg.h5ad")
control_X = np.asarray(hvg[hvg.obs["perturbation"].astype(str) == "control"].X.todense()).astype(np.float32)
rng = np.random.RandomState(0)
cell_idx = rng.choice(control_X.shape[0], 50, replace=False)
cell_batch = torch.tensor(control_X[cell_idx])

true_deltas_matrix = np.stack([deltas_npz[k] for k in deltas_npz.files if k != "genes"])
max_observed_norm = np.linalg.norm(true_deltas_matrix, axis=1).max()

candidates = [
    "_".join(pair)
    for pair in itertools.combinations(vocab_genes, 2)
    if "_".join(pair) not in existing_perturbations and "_".join(pair[::-1]) not in existing_perturbations
]
print(f"Untested candidate gene pairs: {len(candidates)}")

rows = []
for i, candidate in enumerate(candidates):
    genes = candidate.split("_")
    pemb = pair_embedding(emb, embed_dim, genes).astype(np.float32)
    pemb_t = torch.tensor(np.tile(pemb, (cell_batch.shape[0], 1)))
    mean_pred, std_pred = ensemble_predict(models, cell_batch, pemb_t)

    effect_norm = float(np.linalg.norm(mean_pred)) / max_observed_norm
    top_genes = set(np.array(hvg_genes)[np.argsort(-np.abs(mean_pred))[:50]])
    pathway_relevance = len(top_genes & set(found_markers)) / max(len(found_markers), 1)
    uncertainty = float(std_pred.mean())

    rows.append(
        {
            "candidate": candidate,
            "gene_1": genes[0],
            "gene_2": genes[1],
            "effect": min(effect_norm, 1.0),
            "pathway_relevance": pathway_relevance,
            "uncertainty": uncertainty,
        }
    )
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(candidates)} candidates scored")

candidates_df = pd.DataFrame(rows)
ranked = rank_candidates(candidates_df)
ranked["experiment_priority_0_10"] = 10 * ranked["priority"] / ranked["priority"].max()

ranked.to_csv(TABLE_DIR / "virtual_screen_full.csv", index=False)
top20 = ranked.head(20)
top20.to_csv(TABLE_DIR / "virtual_screen_top20.csv", index=False)

print("\nTop 20 prioritized untested gene-pair candidates:")
print(top20[["candidate", "effect", "pathway_relevance", "uncertainty", "experiment_priority_0_10"]].to_string(index=False))
print("Done.")
