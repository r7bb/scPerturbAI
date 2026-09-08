"""Perturbation-level train/test splits with 0/1/2-unseen-gene combo categories.

Unlike a random cell-level split, this holds out entire perturbations so that a
perturbation's cells never appear in both train and test. To measure genuine
generalization to novel gene combinations we additionally hold a subset of
*single*-gene perturbations entirely out of training, which lets two-gene combos
be categorized by how many of their two genes were ever seen individually:

  0/2 unseen: both genes seen as singles in training (combo itself still held out)
  1/2 unseen: only one gene seen as a single in training
  2/2 unseen: neither gene seen as a single in training
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class PerturbationSplit:
    train: list[str]
    test_unseen_single: list[str]
    test_combo_0_unseen: list[str]
    test_combo_1_unseen: list[str]
    test_combo_2_unseen: list[str]
    held_out_genes: set = field(default_factory=set)

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "test_unseen_single": len(self.test_unseen_single),
            "test_combo_0_unseen": len(self.test_combo_0_unseen),
            "test_combo_1_unseen": len(self.test_combo_1_unseen),
            "test_combo_2_unseen": len(self.test_combo_2_unseen),
        }


def build_splits(
    perturbations: list[str],
    held_out_gene_fraction: float = 0.25,
    combo_0_test_fraction: float = 0.3,
    seed: int = 0,
) -> PerturbationSplit:
    rng = random.Random(seed)

    non_control = [p for p in perturbations if p != "control"]
    singles = [p for p in non_control if "_" not in p]
    combos = [p for p in non_control if "_" in p]

    single_genes = sorted(set(singles))
    n_held_out = round(len(single_genes) * held_out_gene_fraction)
    held_out_genes = set(rng.sample(single_genes, n_held_out))

    train_singles = [g for g in singles if g not in held_out_genes]
    test_unseen_single = [g for g in singles if g in held_out_genes]

    combo_0, combo_1, combo_2 = [], [], []
    for combo in combos:
        g1, g2 = combo.split("_")
        n_unseen = (g1 in held_out_genes) + (g2 in held_out_genes)
        if n_unseen == 0:
            combo_0.append(combo)
        elif n_unseen == 1:
            combo_1.append(combo)
        else:
            combo_2.append(combo)

    rng.shuffle(combo_0)
    n_combo_0_test = round(len(combo_0) * combo_0_test_fraction)
    test_combo_0 = combo_0[:n_combo_0_test]
    train_combo_0 = combo_0[n_combo_0_test:]

    train = train_singles + train_combo_0

    return PerturbationSplit(
        train=train,
        test_unseen_single=test_unseen_single,
        test_combo_0_unseen=test_combo_0,
        test_combo_1_unseen=combo_1,
        test_combo_2_unseen=combo_2,
        held_out_genes=held_out_genes,
    )
