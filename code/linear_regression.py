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
        
        X_mat = self._ensure_2d(X)
        y_vec = np.array(y)
        
        # Ensure y is (n_samples, 1) for matrix math
        if y_vec.ndim == 1:
            y_vec = y_vec.reshape(-1, 1)
            
        if X_mat.shape[0] != y_vec.shape[0]:
            raise ValueError(f"Shape mismatch: X has {X_mat.shape[0]} samples, y has {y_vec.shape[0]}.")

        # 2. Augment X with bias column if needed
        X_aug = self._augment_features(X_mat)

        # 3. Setup Regularization Matrix (Identity matrix * lambda)
        n_features_aug = X_aug.shape[1]
        I = np.eye(n_features_aug)
        
        if self.fit_intercept:
            I[0, 0] = 0.0

        # 4. Solve Normal Equation: (X.T @ X + lambda * I) * w = X.T @ y
        XT = X_aug.T
        
        # Calculate X.T @ X + lambda * I
        lhs = (XT @ X_aug) + (self.reg_strength * I)
        
        # Calculate X.T @ y
        rhs = XT @ y_vec

        # Solve for weights w
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices (rare with regularization, but good practice)
            weights = np.linalg.pinv(lhs) @ rhs

        # 5. Extract and store parameters
        weights = weights.flatten() # Flatten back to 1D array for storage

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights

        return self

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
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X_mat = self._ensure_2d(X)

        predictions = X_mat @ self.coef_ + self.intercept_
        
        return predictions

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """
        Optionally prepend a bias column to X.
        """
        if not self.fit_intercept:
            return X
        
        n_samples = X.shape[0]
        # Create column of ones: shape (n_samples, 1)
        ones_col = np.ones((n_samples, 1))
        
        # Stack [1, X] horizontally
        return np.hstack([ones_col, X])


    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        """
        Coerce the input into a 2-D NumPy array of floats.
        """
        X_arr = np.array(X, dtype=float)
        
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        
        # Ensure it is now 2D
        if X_arr.ndim != 2:
            raise ValueError(f"Input must be 2D, but got shape {X_arr.shape}")

        return X_arr


