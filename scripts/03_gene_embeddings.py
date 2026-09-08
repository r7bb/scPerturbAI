"""Build co-expression + perturbation-response gene embeddings and cache them."""
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.gene_embeddings import (
    compute_coexpression_embedding,
    compute_perturbation_response_embedding,
    build_gene_embedding_table,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading full log-normalized matrix (control cells only)...")
full = sc.read_h5ad(DATA_DIR / "processed_full.h5ad")
control = full[full.obs["perturbation"].astype(str) == "control"]
all_genes = control.var_names.tolist()

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)

deltas_npz = np.load(DATA_DIR / "pseudobulk_deltas.npz", allow_pickle=True)
hvg_genes = deltas_npz["genes"].tolist()
deltas = {k: deltas_npz[k] for k in deltas_npz.files if k != "genes"}

train_perts = splits["train"]
train_singles = [p for p in train_perts if "_" not in p]

target_genes = sorted(set(g for p in (train_perts + splits["test_unseen_single"] + splits["test_combo_0_unseen"] + splits["test_combo_1_unseen"] + splits["test_combo_2_unseen"]) for g in p.split("_")))
print(f"Genes needing embeddings: {len(target_genes)}")

all_genes_set = set(control.var_names)
missing_genes = [g for g in target_genes if g not in all_genes_set]
target_genes = [g for g in target_genes if g in all_genes_set]
if missing_genes:
    print(f"WARNING: {len(missing_genes)} perturbation gene symbols not found in expression matrix "
          f"(likely outdated aliases), embedding will fall back to zeros: {missing_genes}")

print("Selecting landmark genes (top 500 by variance among HVGs, all present in full matrix)...")
hvg = sc.read_h5ad(DATA_DIR / "processed_hvg.h5ad")
mean = np.asarray(hvg.X.mean(axis=0)).ravel()
mean_sq = np.asarray(hvg.X.multiply(hvg.X).mean(axis=0)).ravel()
gene_var = mean_sq - mean**2
top_idx = np.argsort(-gene_var)[:500]
landmark_genes = [hvg.var_names[i] for i in top_idx]
landmark_genes = [g for g in landmark_genes if g in all_genes]
print(f"Landmark genes: {len(landmark_genes)}")

print("Computing co-expression embeddings (this touches the 2.7GB matrix once)...")
control_expr = control.X
coexpr = compute_coexpression_embedding(
    control_expr, all_genes, target_genes, landmark_genes, n_components=32, seed=0
)

print("Computing perturbation-response embeddings (training singles only)...")
pert_resp, pert_pca = compute_perturbation_response_embedding(deltas, train_singles, n_components=32, seed=0)
print(f"  {len(pert_resp)} / {len(train_singles)} training singles have delta profiles")

embedding_table = build_gene_embedding_table(coexpr, pert_resp, pert_response_dim=32)

np.savez(
    MODEL_DIR / "gene_embeddings.npz",
    **{gene: vec for gene, vec in embedding_table.items()},
)
np.savez(
    MODEL_DIR / "gene_embeddings_coexpr_only.npz",
    **{gene: np.concatenate([vec, np.zeros(32)]) for gene, vec in coexpr.items()},
)

print("Saved embeddings for", len(embedding_table), "genes to models/gene_embeddings.npz")
print("Example embedding shape:", next(iter(embedding_table.values())).shape)
print("Done.")
