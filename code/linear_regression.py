"""Closed-form linear regression with optional intercept and L2 regularisation."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept: bool = True, reg_strength: float = 0.0) -> None:
        """
        Closed-form linear regression solver with optional L2 regularisation.

        Args:
            fit_intercept (bool): Whether to augment X with a bias column.
            reg_strength (float): Ridge penalty applied to coefficients
                (intercept excluded).
        """
        if reg_strength < 0:
            raise ValueError("reg_strength cannot be negative.")
        self.fit_intercept = fit_intercept
        self.reg_strength = reg_strength
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Solve the normal equations and store weights/intercept.

        Args:
            X (array-like): Training design matrix (n_samples, n_features).
            y (array-like): Target vector (n_samples,) or (n_samples, 1).

        Returns:
            LinearRegression: Fitted estimator (self).

        Raises:
            ValueError: If X and y have different numbers of samples.
        """
        raise NotImplementedError("Implement LinearRegression.fit.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for new samples.

        Args:
            X (array-like): Input matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: 1-D array of predictions.

        Raises:
            RuntimeError: If called before fit.
        """
        raise NotImplementedError("Implement LinearRegression.predict.")

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """
        Optionally prepend a bias column to X.
        """
        raise NotImplementedError("Implement LinearRegression._augment_features.")

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        """
        Coerce the input into a 2-D NumPy array of floats.
        """
        raise NotImplementedError("Implement LinearRegression._ensure_2d.")

