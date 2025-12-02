"""Reusable polynomial feature transformer supporting n-dimensional inputs."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class PolynomialTransformer:
    """
    Generate polynomial feature expansions similar to sklearn's PolynomialFeatures.

    Attributes
    ----------
    degree : int
        Maximum total degree of the polynomial terms.
    include_bias : bool
        Whether to prepend a column of ones.
    n_features_in_ : int | None
        Number of input features seen during fit.
    combinations_ : list[tuple[int, ...]]
        Cached index combinations for generating monomials (excluding the bias term).
    """

    def __init__(self, degree: int = 2, include_bias: bool = True) -> None:
        """
        Create a transformer that expands inputs into polynomial feature space.

        Args:
            degree (int): Maximum total polynomial degree (>= 0).
            include_bias (bool): When True, prepend a constant column of ones.
        """
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        self.degree = degree
        self.include_bias = include_bias
        self.n_features_in_: int | None = None
        self.combinations_: List[Tuple[int, ...]] | None = None

    def fit(self, X: Sequence[Sequence[float]]) -> "PolynomialTransformer":
        """
        Learn the dimensionality of the input and cache index combinations.

        Args:
            X (array-like): Training data with shape (n_samples, n_features).

        Returns:
            PolynomialTransformer: The fitted transformer (self).
        """
        X_arr = self._validate_input(X)

        # 2. Store the number of features
        self.n_features_in_ = X_arr.shape[1]

        # 3. Generate and cache the combinations of indices
        self.combinations_ = self._generate_combinations(self.n_features_in_)

        return self

    def transform(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Apply the learned polynomial expansion to new data.

        Args:
            X (array-like): Input feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Dense design matrix containing all polynomial terms.

        Raises:
            ValueError: If transform is called with a different feature count
                than was seen during fit.
        """
        X_arr = self._validate_input(X)

        # 2. Check if the feature count matches what we saw in fit()
        if self.n_features_in_ is None:
             raise ValueError("The transformer has not been fitted yet.")
        
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, but PolynomialTransformer was fitted with {self.n_features_in_}.")

        n_samples = X_arr.shape[0]
        columns = []

        # 3. Add bias column (column of ones) if requested
        if self.include_bias:
            columns.append(np.ones((n_samples, 1)))

        # 4. Generate polynomial terms based on cached combinations
        # self.combinations_ contains tuples like (0,), (1,), (0, 0), (0, 1)...
        if self.combinations_ is not None:
            for indices in self.combinations_:
                # Start with a column of ones for multiplication
                term = np.ones((n_samples, 1))
                for idx in indices:
                    # Multiply by the column corresponding to the index
                    # Reshape to (n, 1) to ensure we are multiplying columns, not broadcasting incorrectly
                    term = term * X_arr[:, idx].reshape(-1, 1)
                columns.append(term)

        # 5. Stack all columns horizontally
        if not columns:
             # Handle edge case where degree=0 and include_bias=False -> empty matrix
             return np.empty((n_samples, 0))
             
        return np.hstack(columns)

    def fit_transform(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Fit the transformer on X and immediately return the transformed matrix.
        """
        return self.fit(X).transform(X)

    def _generate_combinations(self, n_features: int) -> List[Tuple[int, ...]]:
        """
        Enumerate all index tuples representing monomials up to self.degree.
        """
        combos = []
        # Generate combinations for each degree from 1 up to self.degree
        for d in range(1, self.degree + 1):
            # combinations_with_replacement allows (0, 0) (e.g., x^2)
            for combo in combinations_with_replacement(range(n_features), d):
                combos.append(combo)
        return combos
    

    @staticmethod
    def _validate_input(X: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Ensure X can be interpreted as a 2-D NumPy array of floats.

        Returns:
            np.ndarray: Copy/view of X with shape (n_samples, n_features).

        Raises:
            ValueError: If X cannot be reshaped into 2 dimensions.
        """
        X_arr = np.array(X, dtype=float)
        
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        
        # Ensure it is now 2D
        if X_arr.ndim != 2:
            raise ValueError(f"Input must be 2D, but got shape {X_arr.shape}")

        return X_arr
