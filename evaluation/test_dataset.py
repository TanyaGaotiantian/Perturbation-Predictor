import numpy as np

from evaluation.dataset import RegressionDataset


def test_dataset_shape():
    x = np.random.randn(4, 100)
    y = np.random.randn(4, 4422)
    dataset = RegressionDataset(x, y)

    sample_x, sample_y = dataset[0]

    assert sample_x.shape[0] == 100
    assert sample_y.shape[0] == 4422


def test_dataset_length_matches_samples():
    x = np.random.randn(7, 100)
    y = np.random.randn(7, 4422)
    dataset = RegressionDataset(x, y)

    assert len(dataset) == 7
