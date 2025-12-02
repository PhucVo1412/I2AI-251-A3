"""Reference implementation for regression metrics (Part 1B)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _prepare_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert regression targets/predictions into aligned 1-D float arrays.

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pair of flattened float arrays.

    Raises:
        ValueError: If the arrays have different lengths.
    """

    y_true_arr = np.array(y_true).ravel()
    y_pred_arr = np.array(y_pred).ravel()

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("Arrays do not share the same length")
    
    return y_true_arr, y_pred_arr


def mean_absolute_error(y_true, y_pred) -> float:
    """
    Mean absolute error (MAE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Average absolute deviation between prediction and truth.
    """
    y_true, y_pred = _prepare_inputs(y_true,y_pred)
    n = len(y_true)

    if n==0:
        return 0.0
    
    return 1/n * sum(abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred) -> float:
    """
    Mean squared error (MSE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Average squared deviation between prediction and truth.
    """
    y_true, y_pred = _prepare_inputs(y_true,y_pred)
    n = len(y_true)

    if n==0:
        return 0.0
    
    return 1/n * sum((y_true - y_pred)**2)


def root_mean_squared_error(y_true, y_pred) -> float:
    """
    Root mean squared error (RMSE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Square root of the mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true,y_pred))


def r2_score(y_true, y_pred) -> float:
    """
    Coefficient of determination (R²).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: R² score, 1.0 for perfect predictions.
    """
    y_true, y_pred = _prepare_inputs(y_true,y_pred)
    residual_ss = sum((y_true - y_pred)**2)
    total_ss = sum(((y_true - np.mean(y_true)) **2))
    return 1 - residual_ss/total_ss


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """
    Aggregate common regression metrics into a dictionary.

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        Dict[str, float]: Keys "mae", "mse", "rmse", and "r2".
    """
    report = {}
    report["mae"] = mean_absolute_error(y_true, y_pred)
    report["mse"] = mean_squared_error(y_true, y_pred)
    report["rmse"] = root_mean_squared_error(y_true, y_pred)
    report["r2"] = r2_score(y_true, y_pred)

    return report
    

