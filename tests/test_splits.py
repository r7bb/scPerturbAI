from src.data.splits import build_splits


def test_build_splits_categorizes_combos_by_unseen_gene_count():
    perturbations = [
        "control",
        "A", "B", "C", "D",  # singles
        "A_B",  # both seen (if neither A nor B held out)
        "C_D",
    ]
    split = build_splits(perturbations, held_out_gene_fraction=0.5, combo_0_test_fraction=1.0, seed=1)

    all_test_combos = split.test_combo_0_unseen + split.test_combo_1_unseen + split.test_combo_2_unseen
    assert set(all_test_combos) == {"A_B", "C_D"}
    assert set(split.train) | set(split.test_unseen_single) == {"A", "B", "C", "D"} or set(split.train) <= {"A", "B", "C", "D"}

    for combo in split.test_combo_2_unseen:
        genes = combo.split("_")
        assert all(g in split.held_out_genes for g in genes)
    for combo in split.test_combo_0_unseen:
        genes = combo.split("_")
        assert all(g not in split.held_out_genes for g in genes)


def test_build_splits_no_held_out_genes_puts_all_combos_in_0_unseen_pool():
    perturbations = ["control", "A", "B", "A_B"]
    split = build_splits(perturbations, held_out_gene_fraction=0.0, combo_0_test_fraction=0.0, seed=0)
    assert split.held_out_genes == set()
    assert split.test_unseen_single == []
    assert split.test_combo_1_unseen == []
    assert split.test_combo_2_unseen == []
    assert "A_B" in split.train
