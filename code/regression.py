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
    # 1. Convert to numpy array
    arr = np.array(vector)

    # 2. Handle 1D arrays -> reshape to (n, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    
    # 3. Handle 2D arrays -> ensure second dimension is 1
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr
        else:
            raise ValueError(f"Expected single column, got shape {arr.shape}")
            
    # 4. Reject anything else (scalar, 3D+, etc.)
    raise ValueError(f"Input must be 1D or 2D column vector. Got dimension {arr.ndim}")


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
    if not features:
        raise ValueError("No features provided to stack.")

    # 1. Ensure all inputs are columns
    processed_features = [_ensure_column(f) for f in features]
    
    # 2. Check lengths match
    n_samples = processed_features[0].shape[0]
    for i, feat in enumerate(processed_features):
        if feat.shape[0] != n_samples:
            raise ValueError(f"Feature at index {i} has length {feat.shape[0]}, expected {n_samples}")

    # 3. Stack horizontally 
    return np.hstack(processed_features)


def polynomial_features(x, degree: int) -> np.ndarray:
    """
    Create polynomial design matrix for a single predictor.

    Args:
        x (array-like): Predictor values.
        degree (int): Maximum polynomial degree (>= 0).

    Returns:
        np.ndarray: Polynomial feature matrix including bias column.
    """
    x_col = _ensure_column(x)
    
    # 2. Construct transformer with bias
    # include_bias=True ensures the first column is ones
    transformer = PolynomialTransformer(degree=degree, include_bias=True)
    
    # 3. Fit and transform 
    return transformer.fit_transform(x_col)

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
    X_poly = polynomial_features(x, degree)
    
    # 2. Instantiate LinearRegression 
    # fit_intercept=False because X_poly already has a bias column (ones)
    model = LinearRegression(fit_intercept=False)
    
    # 3. Fit the model 
    # Note: learning_rate and epochs are ignored as LinearRegression uses closed-form solution
    model.fit(X_poly, y)
    
    # 4. Return learned weights
    # Assuming LinearRegression stores weights in .coef_ (including the bias weight since fit_intercept=False)
    return model.coef_

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
    degree = len(weights) - 1
    
    # 2. Rebuild features [cite: 302]
    X_poly = polynomial_features(x, degree)
    
    # 3. Compute predictions via matrix multiplication 
    # Ensure weights are a column vector for dot product, or use simple @ if shapes align
    return X_poly @ weights


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
    X = _stack_features(x1, x2)
    
    # 2. Transform to degree 2 interaction terms 
    # Basis: [1, x1, x2, x1^2, x1x2, x2^2]
    transformer = PolynomialTransformer(degree=2, include_bias=True)
    X_surf = transformer.fit_transform(X)
    
    # 3. Fit LinearRegression [cite: 308]
    model = LinearRegression(fit_intercept=False)
    model.fit(X_surf, y)
    
    return model.coef_


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
    X = _stack_features(x1, x2)
    
    # 2. Transform (hardcoded degree=2 for surface regression) 
    transformer = PolynomialTransformer(degree=2, include_bias=True)
    # We must call fit_transform (or fit then transform) to initialize the transformer's internal state
    X_surf = transformer.fit_transform(X)
    
    # 3. Compute predictions
    return X_surf @ weights
