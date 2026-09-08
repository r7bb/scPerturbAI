"""Train the deep learning perturbation model: control cell + gene-pair embedding -> HVG delta."""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.gene_embeddings import pair_embedding
from src.models.perturbation_model import PerturbationModel
from src.evaluation.metrics import evaluate_predictions

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TABLE_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", default="gene_embeddings.npz")
parser.add_argument("--tag", default="full")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-cells-per-pert", type=int, default=200)
parser.add_argument("--save-model", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print("Loading data...")
with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)

deltas_npz = np.load(DATA_DIR / "pseudobulk_deltas.npz", allow_pickle=True)
hvg_genes = deltas_npz["genes"].tolist()
true_deltas = {k: deltas_npz[k].astype(np.float32) for k in deltas_npz.files if k != "genes"}
n_hvg = len(hvg_genes)

emb = dict(np.load(MODEL_DIR / args.embedding))
embed_dim = next(iter(emb.values())).shape[0]
pair_dim = embed_dim * 2

hvg = sc.read_h5ad(DATA_DIR / "processed_hvg.h5ad")
control_X = np.asarray(hvg[hvg.obs["perturbation"].astype(str) == "control"].X.todense()).astype(np.float32)
print(f"Control cells: {control_X.shape}")

rng = np.random.RandomState(args.seed)
all_train_perts = splits["train"]
rng.shuffle(all_train_perts)
n_val = max(5, round(len(all_train_perts) * 0.15))
val_perts = all_train_perts[:n_val]
fit_perts = all_train_perts[n_val:]
print(f"Fit perturbations: {len(fit_perts)}, validation perturbations: {len(val_perts)}")


class PerturbationDataset(Dataset):
    def __init__(self, perturbations, max_cells):
        self.samples = []
        for pert in perturbations:
            if pert not in true_deltas:
                continue
            n = min(max_cells, control_X.shape[0])
            idx = rng.choice(control_X.shape[0], n, replace=False)
            pemb = pair_embedding(emb, embed_dim, pert.split("_")).astype(np.float32)
            for i in idx:
                self.samples.append((i, pemb, true_deltas[pert]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        cell_idx, pemb, target = self.samples[i]
        return control_X[cell_idx], pemb, target


train_ds = PerturbationDataset(fit_perts, args.max_cells_per_pert)
val_ds = PerturbationDataset(val_perts, 50)
print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = PerturbationModel(n_genes=n_hvg, embed_dim=pair_dim, hidden_dim=256)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss()

best_val = float("inf")
best_state = None
patience, bad_epochs = 10, 0
history = []

print("Training...")
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    for cell, pemb, target in train_loader:
        optimizer.zero_grad()
        pred = model(cell, pemb)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * cell.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for cell, pemb, target in val_loader:
            pred = model(cell, pemb)
            val_loss += loss_fn(pred, target).item() * cell.size(0)
    val_loss /= max(len(val_ds), 1)

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val - 1e-5:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(TABLE_DIR / f"training_curve_{args.tag}.csv", index=False)

if args.save_model:
    torch.save(
        {"state_dict": model.state_dict(), "n_hvg": n_hvg, "pair_dim": pair_dim, "embed_dim": embed_dim},
        MODEL_DIR / f"perturbation_model_{args.tag}.pt",
    )

print("Evaluating on all categories...")
model.eval()

mean_control = control_X.mean(axis=0, keepdims=True)


def predict_delta(pert: str) -> np.ndarray:
    pemb = pair_embedding(emb, embed_dim, pert.split("_")).astype(np.float32)
    n = min(100, control_X.shape[0])
    idx = rng.choice(control_X.shape[0], n, replace=False)
    with torch.no_grad():
        cell = torch.tensor(control_X[idx])
        pemb_t = torch.tensor(np.tile(pemb, (n, 1)))
        pred = model(cell, pemb_t).numpy()
    return pred.mean(axis=0)


categories = {
    "train (seen, sanity check)": fit_perts,
    "val (held-out train genes' combo)": val_perts,
    "test_unseen_single": splits["test_unseen_single"],
    "test_combo_0_unseen": splits["test_combo_0_unseen"],
    "test_combo_1_unseen": splits["test_combo_1_unseen"],
    "test_combo_2_unseen": splits["test_combo_2_unseen"],
}
all_perts = sorted(set(p for perts in categories.values() for p in perts if p in true_deltas))
preds = {p: predict_delta(p) for p in all_perts}

_, summary = evaluate_predictions(true_deltas, preds, categories)
summary = summary.reset_index()
summary.insert(0, "model", f"MLP ({args.tag})")
summary.to_csv(TABLE_DIR / f"mlp_metrics_{args.tag}.csv", index=False)
print(summary.to_string(index=False))
print("Done.")
