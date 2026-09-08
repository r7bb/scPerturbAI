"""Load, QC, normalize, HVG-select, and split the Norman Perturb-seq dataset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import load_dataset, run_qc, normalize, pseudobulk_deltas
from src.data.splits import build_splits

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

print("Loading dataset...")
adata = load_dataset(str(DATA_DIR / "norman.h5ad"))
print(adata)

print("Running QC...")
adata = run_qc(adata)
print(f"After QC: {adata.n_obs} cells")

print("Normalizing + selecting HVGs...")
full, hvg = normalize(adata, n_top_genes=2000)
print(f"HVG matrix: {hvg.shape}")

print("Building perturbation-level splits...")
perturbations = adata.obs["perturbation"].astype(str).unique().tolist()
split = build_splits(perturbations, held_out_gene_fraction=0.25, combo_0_test_fraction=0.3, seed=0)
print("Split summary:", split.summary())
print("Held-out genes:", sorted(split.held_out_genes))

print("Computing pseudobulk deltas (per perturbation, HVG space)...")
deltas = pseudobulk_deltas(hvg)

print("Saving artifacts...")
hvg.write_h5ad(DATA_DIR / "processed_hvg.h5ad")
full.write_h5ad(DATA_DIR / "processed_full.h5ad")

import numpy as np
np.savez(
    DATA_DIR / "pseudobulk_deltas.npz",
    genes=np.array(hvg.var_names),
    **{k: v for k, v in deltas.items()},
)

with open(DATA_DIR / "splits.json", "w") as f:
    json.dump(
        {
            "train": split.train,
            "test_unseen_single": split.test_unseen_single,
            "test_combo_0_unseen": split.test_combo_0_unseen,
            "test_combo_1_unseen": split.test_combo_1_unseen,
            "test_combo_2_unseen": split.test_combo_2_unseen,
            "held_out_genes": sorted(split.held_out_genes),
        },
        f,
        indent=2,
    )

print("Done.")
