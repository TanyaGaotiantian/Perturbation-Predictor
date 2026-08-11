import torch

from evaluation.model import MLP


def test_mlp_output():
    model = MLP(input_dim=100, output_dim=4422)
    x = torch.randn(8, 100)

    pred = model(x)

    assert pred.shape == (8, 4422)


def test_mlp_is_batchable():
    model = MLP(input_dim=100, output_dim=4422)
    x = torch.randn(1, 100)

    pred = model(x)

    assert pred.shape == (1, 4422)
