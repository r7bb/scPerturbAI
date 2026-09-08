import torch

from src.models.perturbation_model import PerturbationModel, load_model
from src.screening.uncertainty import ensemble_predict


def test_perturbation_model_output_shape():
    n_genes, embed_dim, batch = 100, 64, 4
    model = PerturbationModel(n_genes=n_genes, embed_dim=embed_dim)

    control_expression = torch.randn(batch, n_genes)
    perturbation_embedding = torch.randn(batch, embed_dim)

    output = model(control_expression, perturbation_embedding)

    assert output.shape == (batch, n_genes)


def test_load_model_roundtrip(tmp_path):
    n_genes, embed_dim = 20, 8
    model = PerturbationModel(n_genes=n_genes, embed_dim=embed_dim)
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "n_hvg": n_genes, "pair_dim": embed_dim, "embed_dim": embed_dim},
        ckpt_path,
    )

    loaded, checkpoint = load_model(ckpt_path)
    assert checkpoint["n_hvg"] == n_genes

    x = torch.randn(3, n_genes)
    p = torch.randn(3, embed_dim)
    torch.testing.assert_close(model(x, p), loaded(x, p))


def test_ensemble_predict_returns_mean_and_std_across_models():
    n_genes, embed_dim = 10, 4
    torch.manual_seed(0)
    models = [PerturbationModel(n_genes=n_genes, embed_dim=embed_dim) for _ in range(3)]

    cells = torch.randn(5, n_genes)
    pert_emb = torch.randn(5, embed_dim)
    mean_pred, std_pred = ensemble_predict(models, cells, pert_emb)

    assert mean_pred.shape == (n_genes,)
    assert std_pred.shape == (n_genes,)
    assert (std_pred >= 0).all()


def test_ensemble_predict_zero_std_for_identical_models():
    n_genes, embed_dim = 10, 4
    torch.manual_seed(0)
    model = PerturbationModel(n_genes=n_genes, embed_dim=embed_dim)
    models = [model, model, model]

    cells = torch.randn(5, n_genes)
    pert_emb = torch.randn(5, embed_dim)
    _, std_pred = ensemble_predict(models, cells, pert_emb)

    assert (std_pred < 1e-6).all()
