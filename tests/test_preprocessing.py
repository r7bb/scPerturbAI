import anndata as ad
import numpy as np
import pytest

from src.data.preprocessing import normalize, perturbation_genes, pseudobulk_deltas, run_qc


def make_synthetic_adata(n_cells=200, n_genes=50, seed=0):
    rng = np.random.RandomState(seed)
    counts = rng.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)
    obs = {
        "ngenes": (counts > 0).sum(axis=1),
        "ncounts": counts.sum(axis=1),
        "percent_mito": rng.uniform(0, 10, size=n_cells),
        "perturbation": rng.choice(["control", "GENEA", "GENEB", "GENEA_GENEB"], size=n_cells),
    }
    return ad.AnnData(X=counts, obs=obs)


def test_run_qc_filters_on_thresholds():
    adata = make_synthetic_adata()
    adata.obs.loc[adata.obs.index[0], "ngenes"] = 1
    adata.obs.loc[adata.obs.index[0], "percent_mito"] = 50.0
    filtered = run_qc(adata, min_genes=5, max_percent_mito=20.0)
    assert filtered.n_obs < adata.n_obs
    assert (filtered.obs["ngenes"] >= 5).all()
    assert (filtered.obs["percent_mito"] <= 20.0).all()


def test_normalize_returns_full_and_hvg_views():
    adata = make_synthetic_adata(n_genes=50)
    full, hvg = normalize(adata, n_top_genes=10)
    assert full.n_vars == 50
    # scanpy's dispersion-cutoff HVG selection can return a few more genes than
    # requested when there are ties (common with small/synthetic count data).
    assert 0 < hvg.n_vars <= 20
    assert full.n_obs == hvg.n_obs == adata.n_obs


@pytest.mark.parametrize(
    "label,expected",
    [("control", []), ("GENEA", ["GENEA"]), ("GENEA_GENEB", ["GENEA", "GENEB"])],
)
def test_perturbation_genes(label, expected):
    assert perturbation_genes(label) == expected


def test_pseudobulk_deltas_excludes_control_and_matches_hvg_width():
    adata = make_synthetic_adata(n_genes=50)
    _, hvg = normalize(adata, n_top_genes=10)
    deltas = pseudobulk_deltas(hvg)
    assert "control" not in deltas
    assert len(deltas) > 0
    assert all(v.shape == (hvg.n_vars,) for v in deltas.values())
