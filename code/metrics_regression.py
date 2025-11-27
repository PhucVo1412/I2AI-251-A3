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
    raise NotImplementedError("Implement _prepare_inputs.")


def mean_absolute_error(y_true, y_pred) -> float:
    """
    Mean absolute error (MAE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Average absolute deviation between prediction and truth.
    """
    raise NotImplementedError("Implement mean_absolute_error.")


def mean_squared_error(y_true, y_pred) -> float:
    """
    Mean squared error (MSE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Average squared deviation between prediction and truth.
    """
    raise NotImplementedError("Implement mean_squared_error.")


def root_mean_squared_error(y_true, y_pred) -> float:
    """
    Root mean squared error (RMSE).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Square root of the mean squared error.
    """
    raise NotImplementedError("Implement root_mean_squared_error.")


def r2_score(y_true, y_pred) -> float:
    """
    Coefficient of determination (R²).

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: R² score, 1.0 for perfect predictions.
    """
    raise NotImplementedError("Implement r2_score.")


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """
    Aggregate common regression metrics into a dictionary.

    Args:
        y_true (array-like): Ground-truth values.
        y_pred (array-like): Predicted values.

    Returns:
        Dict[str, float]: Keys "mae", "mse", "rmse", and "r2".
    """
    raise NotImplementedError("Implement regression_report.")

