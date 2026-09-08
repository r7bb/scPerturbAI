"""Exploratory analysis: PCA/UMAP, perturbation frequencies, E-distance ranking."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.biology.differential_expression import e_distance

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sc.settings.figdir = str(FIG_DIR)

print("Loading HVG matrix...")
hvg = sc.read_h5ad(DATA_DIR / "processed_hvg.h5ad")
print(hvg)

print("Perturbation cell-count distribution...")
counts = hvg.obs["perturbation"].astype(str).value_counts()
fig, ax = plt.subplots(figsize=(10, 4))
counts.sort_values(ascending=False).plot(kind="bar", ax=ax, width=0.8)
ax.set_xticklabels([])
ax.set_xlabel("Perturbation (237 conditions, sorted by frequency)")
ax.set_ylabel("Number of cells")
ax.set_title("Cells per perturbation condition")
fig.tight_layout()
fig.savefig(FIG_DIR / "perturbation_counts.png", dpi=150)
plt.close(fig)
counts.to_csv(TABLE_DIR / "perturbation_cell_counts.csv", header=["n_cells"])

print("Running PCA on full dataset...")
sc.pp.scale(hvg, max_value=10)
sc.tl.pca(hvg, n_comps=50, svd_solver="arpack")

print("Subsampling for UMAP...")
rng = np.random.RandomState(0)
sub_idx = rng.choice(hvg.n_obs, size=min(20000, hvg.n_obs), replace=False)
sub = hvg[sub_idx].copy()
sc.pp.neighbors(sub, n_pcs=50)
sc.tl.umap(sub)

sub.obs["is_control"] = np.where(sub.obs["perturbation"].astype(str) == "control", "control", "perturbed")
fig = sc.pl.umap(sub, color="is_control", show=False, return_fig=True, title="Control vs. perturbed cells")
fig.savefig(FIG_DIR / "umap_control_vs_perturbed.png", dpi=150, bbox_inches="tight")
plt.close(fig)

top_perts = counts.drop("control").head(8).index.tolist()
sub.obs["highlight_pert"] = np.where(sub.obs["perturbation"].astype(str).isin(top_perts), sub.obs["perturbation"].astype(str), "other")
fig = sc.pl.umap(sub, color="highlight_pert", show=False, return_fig=True, title="Top 8 most frequent perturbations")
fig.savefig(FIG_DIR / "umap_top_perturbations.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Computing E-distance per perturbation (in PCA space)...")
pca_embedding = hvg.obsm["X_pca"]
perturbation_labels = hvg.obs["perturbation"].astype(str).values
control_idx = np.where(perturbation_labels == "control")[0]

e_distances = {}
unique_perts = [p for p in np.unique(perturbation_labels) if p != "control"]
for i, pert in enumerate(unique_perts):
    pert_idx = np.where(perturbation_labels == pert)[0]
    e_distances[pert] = e_distance(pca_embedding, pert_idx, control_idx, max_cells=300, seed=0)
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(unique_perts)} perturbations done")

e_dist_df = pd.DataFrame(sorted(e_distances.items(), key=lambda x: -x[1]), columns=["perturbation", "e_distance"])
e_dist_df.to_csv(TABLE_DIR / "e_distance.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 8))
top20 = e_dist_df.head(20).sort_values("e_distance")
ax.barh(top20["perturbation"], top20["e_distance"])
ax.set_xlabel("E-distance from control (PCA space)")
ax.set_title("Top 20 strongest transcriptional perturbations")
fig.tight_layout()
fig.savefig(FIG_DIR / "top_e_distance.png", dpi=150)
plt.close(fig)

print("Top 10 strongest perturbations:")
print(e_dist_df.head(10).to_string(index=False))
print("Done.")
