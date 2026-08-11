from __future__ import annotations

import numpy as np


def evaluate(y_true, y_pred):
    """Return RMSE and R2 for regression predictions."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    diff = y_true_arr - y_pred_arr
    rmse = float(np.sqrt(np.mean(np.square(diff))))

    y_true_mean = np.mean(y_true_arr)
    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(y_true_arr - y_true_mean)))

    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    return {"RMSE": rmse, "R2": float(r2)}
