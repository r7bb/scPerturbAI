"""Biological representations of CRISPR target genes.

Two complementary, fully data-driven embedding sources (no external knowledge-graph
download required):

1. Co-expression embedding: every gene's correlation profile against a fixed set of
   landmark (highly-variable) genes, computed on control cells, PCA-reduced. This acts
   as a lightweight, self-contained proxy for a gene-gene functional-similarity network
   (the same intuition behind STRING/co-expression-network priors) and is available for
   *any* measured gene, including ones never individually perturbed.

2. Perturbation-response embedding: the HVG-space pseudobulk delta induced by a gene's
   own single-gene perturbation, PCA-reduced. Only available for genes with single-gene
   perturbation data in the *training* split — held out for any gene we want to treat as
   genuinely "unseen" (see src/data/splits.py). Missing entries are zero-filled.

The ablation between "co-expression only" and "co-expression + perturbation-response"
is the biological-knowledge ablation described in the project plan (does knowing how a
gene behaves under perturbation help beyond static expression structure), implemented
without any external KG/LLM dependency.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def compute_coexpression_embedding(
    expression: np.ndarray,
    all_genes: list[str],
    target_genes: list[str],
    landmark_genes: list[str],
    n_components: int = 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """PCA-reduced correlation profile of each target gene against landmark genes.

    `expression` is a (n_control_cells, n_all_genes) log-normalized dense/sparse array
    aligned with `all_genes`.
    """
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    landmark_idx = [gene_to_idx[g] for g in landmark_genes]
    landmark_expr = np.asarray(expression[:, landmark_idx].todense() if hasattr(expression, "todense") else expression[:, landmark_idx])

    landmark_z = (landmark_expr - landmark_expr.mean(axis=0)) / (landmark_expr.std(axis=0) + 1e-8)

    profiles = []
    for gene in target_genes:
        idx = gene_to_idx[gene]
        gene_expr = np.asarray(expression[:, idx].todense() if hasattr(expression, "todense") else expression[:, idx]).ravel()
        gene_z = (gene_expr - gene_expr.mean()) / (gene_expr.std() + 1e-8)
        corr = (landmark_z * gene_z[:, None]).mean(axis=0)
        profiles.append(corr)
    profiles = np.nan_to_num(np.stack(profiles))

    n_components = min(n_components, profiles.shape[0], profiles.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
    reduced = pca.fit_transform(profiles)
    return {gene: reduced[i] for i, gene in enumerate(target_genes)}


def compute_perturbation_response_embedding(
    deltas: dict[str, np.ndarray],
    train_single_genes: list[str],
    n_components: int = 32,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], PCA]:
    """PCA-reduced pseudobulk delta profile, fit only on training single-gene perturbations."""
    genes = [g for g in train_single_genes if g in deltas]
    matrix = np.stack([deltas[g] for g in genes])
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
    reduced = pca.fit_transform(matrix)
    return {gene: reduced[i] for i, gene in enumerate(genes)}, pca


def build_gene_embedding_table(
    coexpression: dict[str, np.ndarray],
    perturbation_response: dict[str, np.ndarray],
    pert_response_dim: int,
) -> dict[str, np.ndarray]:
    """Concatenate [co-expression | perturbation-response-or-zeros] per gene."""
    table = {}
    for gene, coexpr_vec in coexpression.items():
        pert_vec = perturbation_response.get(gene, np.zeros(pert_response_dim))
        table[gene] = np.concatenate([coexpr_vec, pert_vec])
    return table


def pair_embedding(gene_embeddings: dict[str, np.ndarray], embed_dim: int, genes: list[str]) -> np.ndarray:
    """Symmetric [sum | abs-diff] representation for a perturbation of 1 or 2 genes."""
    vecs = [gene_embeddings.get(g, np.zeros(embed_dim)) for g in genes]
    if len(vecs) == 1:
        vecs.append(np.zeros(embed_dim))
    a, b = vecs
    return np.concatenate([a + b, np.abs(a - b)])
