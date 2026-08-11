from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is expected for training/tests
    torch = None
    nn = None


class MLP(nn.Module):
    def __init__(self, input_dim: int = 100, output_dim: int = 4422, hidden_dim: int = 256):
        if nn is None:
            raise ImportError("torch is required to use MLP")

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)
