from __future__ import annotations

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is expected for training/tests
    torch = None
    Dataset = object


class RegressionDataset(Dataset):
    def __init__(self, x, y):
        if torch is None:
            raise ImportError("torch is required to use RegressionDataset")

        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

