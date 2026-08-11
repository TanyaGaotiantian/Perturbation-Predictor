import numpy as np

from evaluation.evaluator import evaluate


def test_perfect_prediction():
    y_true = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    y_pred = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    result = evaluate(y_true, y_pred)

    assert result["RMSE"] == 0
    assert result["R2"] == 1


def test_shifted_prediction_changes_metrics():
    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[2, 3], [4, 5]])

    result = evaluate(y_true, y_pred)

    assert result["RMSE"] > 0
    assert result["R2"] < 1
