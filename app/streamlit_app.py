"""VirtualCRISPR demo: pick a gene pair, see predicted expression changes, pathways, and confidence."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.gene_embeddings import pair_embedding
from src.models.perturbation_model import load_model
from src.screening.uncertainty import ensemble_predict
from src.biology.pathway_analysis import enrich_pathways

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "results" / "tables"
SEEDS = [11, 22, 33, 44, 55]

st.set_page_config(page_title="VirtualCRISPR", layout="wide")


@st.cache_resource
def load_artifacts():
    app_artifacts = np.load(MODEL_DIR / "app_artifacts.npz", allow_pickle=True)
    hvg_genes = app_artifacts["hvg_genes"].tolist()
    control_X = app_artifacts["control_reference"].astype(np.float32)

    emb = dict(np.load(MODEL_DIR / "gene_embeddings.npz"))
    embed_dim = next(iter(emb.values())).shape[0]
    models = [load_model(MODEL_DIR / f"perturbation_model_ens{s}.pt")[0] for s in SEEDS]

    with open(DATA_DIR / "splits.json") as f:
        import json

        splits = json.load(f)
    return hvg_genes, emb, embed_dim, models, control_X, splits


try:
    hvg_genes, emb, embed_dim, models, control_X, splits = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

st.title("VirtualCRISPR")
st.caption("AI-powered genetic perturbation screen — Norman et al. Perturb-seq, K562 cells")

if not artifacts_loaded:
    st.warning(
        "Model artifacts not found in `models/`. If you cloned the repo, they should be "
        "committed already — check `models/app_artifacts.npz`, `models/gene_embeddings.npz`, "
        "and `models/perturbation_model_ens{11,22,33,44,55}.pt`. To regenerate from scratch, run "
        "`scripts/01_preprocess.py` → `03_gene_embeddings.py` → `05_train_model.py --tag ens{seed} "
        "--seed {seed} --save-model` for each seed → `10_export_app_artifacts.py`."
    )
    st.stop()

gene_vocab = sorted(emb.keys())
category_of = {}
for cat in ["train", "test_unseen_single", "test_combo_0_unseen", "test_combo_1_unseen", "test_combo_2_unseen"]:
    for p in splits[cat]:
        for g in p.split("_"):
            category_of.setdefault(g, cat)

col1, col2, col3 = st.columns(3)
gene_a = col1.selectbox("Target Gene 1", gene_vocab, index=gene_vocab.index("KLF1") if "KLF1" in gene_vocab else 0)
gene_b_options = ["(none — single-gene perturbation)"] + gene_vocab
gene_b = col2.selectbox("Target Gene 2 (optional)", gene_b_options, index=0)
col3.metric("Cell model", "K562")

for g in [gene_a] + ([gene_b] if gene_b != gene_b_options[0] else []):
    cat = category_of.get(g, "unknown")
    tag = {
        "train": "seen in training",
        "test_unseen_single": "held out as unseen single",
    }.get(cat, cat)
    st.caption(f"{g}: {tag}")

if st.button("Predict Perturbation", type="primary"):
    genes = [gene_a] if gene_b == gene_b_options[0] else [gene_a, gene_b]
    label = "_".join(genes)

    pemb = pair_embedding(emb, embed_dim, genes).astype(np.float32)
    rng = np.random.RandomState(0)
    idx = rng.choice(control_X.shape[0], 100, replace=False)
    cell_batch = torch.tensor(control_X[idx])
    pemb_t = torch.tensor(np.tile(pemb, (cell_batch.shape[0], 1)))

    with st.spinner("Running ensemble prediction..."):
        mean_pred, std_pred = ensemble_predict(models, cell_batch, pemb_t)

    top_idx = np.argsort(-np.abs(mean_pred))[:15]
    top_genes = [hvg_genes[i] for i in top_idx]
    top_values = mean_pred[top_idx]
    confidence = 1.0 / (1.0 + std_pred[top_idx].mean())

    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Predicted top affected genes: {label}")
        chart_df = pd.DataFrame({"gene": top_genes, "predicted_delta": top_values}).set_index("gene")
        st.bar_chart(chart_df)

    with right:
        st.metric("Model confidence", f"{confidence * 100:.0f}%")
        st.metric("Max |predicted delta|", f"{np.abs(mean_pred).max():.2f}")
        st.metric("Mean ensemble uncertainty (std)", f"{std_pred.mean():.4f}")

    st.subheader("Pathway enrichment of predicted top genes")
    with st.spinner("Querying Enrichr..."):
        try:
            pathways = enrich_pathways(top_genes)
            st.dataframe(pathways[["Term", "Overlap", "Adjusted P-value"]].head(10), width="stretch")
        except Exception as e:
            st.info(f"Pathway enrichment unavailable right now ({e}).")

st.divider()
st.subheader("Top prioritized untested gene pairs (precomputed virtual screen)")
screen_path = TABLE_DIR / "virtual_screen_top20.csv"
if screen_path.exists():
    screen_df = pd.read_csv(screen_path)
    st.dataframe(
        screen_df[["candidate", "effect", "pathway_relevance", "uncertainty", "experiment_priority_0_10"]],
        width="stretch",
    )
else:
    st.info("Run scripts/09_virtual_screen.py to populate this table.")

st.divider()
st.caption(
    "Limitations: predictions are based on one experimental system (K562, CRISPRa, one Perturb-seq screen) "
    "and approximate the population-level (pseudobulk) transcriptional shift, not per-cell heterogeneity. "
    "This is an experiment-prioritization tool, not a substitute for wet-lab validation."
)
