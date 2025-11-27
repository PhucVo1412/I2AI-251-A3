"""Reference implementations for Part 2 – non-linear regression."""

from __future__ import annotations

import numpy as np

from linear_regression import LinearRegression
from polynomial_transformer import PolynomialTransformer


def _ensure_column(vector) -> np.ndarray:
    """
    Convert a 1-D array-like input into a single-column matrix.

    Args:
        vector (array-like): Input values.

    Returns:
        np.ndarray: Shape (n_samples, 1).

    Raises:
        ValueError: If the input cannot be coerced into a column vector.
    """
    raise NotImplementedError("Implement _ensure_column.")


def _stack_features(*features) -> np.ndarray:
    """
    Stack multiple feature vectors into a single 2-D array.

    Args:
        *features (array-like): Vectors of equal length.

    Returns:
        np.ndarray: Matrix whose columns correspond to the input vectors.

    Raises:
        ValueError: If feature vectors have different lengths.
    """
    raise NotImplementedError("Implement _stack_features.")


def polynomial_features(x, degree: int) -> np.ndarray:
    """
    Create polynomial design matrix for a single predictor.

    Args:
        x (array-like): Predictor values.
        degree (int): Maximum polynomial degree (>= 0).

    Returns:
        np.ndarray: Polynomial feature matrix including bias column.
    """
    raise NotImplementedError("Implement polynomial_features.")


def fit_polynomial_regression(
    x,
    y,
    degree: int = 2,
    learning_rate: float = 0.01,
    epochs: int = 2000,
) -> np.ndarray:
    """
    Fit polynomial regression coefficients via closed-form least squares.

    Args:
        x (array-like): Predictor values.
        y (array-like): Target values.
        degree (int): Polynomial degree.
        learning_rate (float): Ignored; kept for parity with student API.
        epochs (int): Ignored; kept for parity with student API.

    Returns:
        np.ndarray: Learned weights (including bias).
    """
    raise NotImplementedError("Implement fit_polynomial_regression.")


def predict_polynomial(
    x,
    weights,
) -> np.ndarray:
    """
    Evaluate a polynomial model at the provided inputs.

    Args:
        x (array-like): Predictor values.
        weights (array-like): Weight vector including bias.

    Returns:
        np.ndarray: Predicted responses.
    """
    raise NotImplementedError("Implement predict_polynomial.")


def fit_surface_regression(
    x1,
    x2,
    y,
    learning_rate: float = 0.01,
    epochs: int = 2500,
) -> np.ndarray:
    """
    Fit a quadratic surface regression with two predictors.

    Args:
        x1 (array-like): First predictor.
        x2 (array-like): Second predictor.
        y (array-like): Target values.
        learning_rate (float): Ignored; API compatibility only.
        epochs (int): Ignored; API compatibility only.

    Returns:
        np.ndarray: Learned weight vector.
    """
    raise NotImplementedError("Implement fit_surface_regression.")


def predict_surface(
    x1,
    x2,
    weights,
) -> np.ndarray:
    """
    Predict outputs from a quadratic surface regression model.

    Args:
        x1 (array-like): First predictor.
        x2 (array-like): Second predictor.
        weights (array-like): Weight vector learned by fit_surface_regression.

    Returns:
        np.ndarray: Predicted responses.
    """
    raise NotImplementedError("Implement predict_surface.")

