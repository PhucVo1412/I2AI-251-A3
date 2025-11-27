import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add code directory to path (following bayesnet pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import regression


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


def load_regression_1d():
    df = pd.read_csv(DATA_DIR / "regression_1d.csv")
    return df["x"].to_numpy(), df["y"].to_numpy()


def load_regression_2d():
    df = pd.read_csv(DATA_DIR / "regression_2d.csv")
    return df["x1"].to_numpy(), df["x2"].to_numpy(), df["y"].to_numpy()


def load_expected_weights(filename: str) -> np.ndarray:
    df = pd.read_csv(EXPECTED_DIR / filename).sort_values("index")
    return df["weight"].to_numpy()


def test_polynomial_features_shape():
    x, _ = load_regression_1d()
    design = regression.polynomial_features(x, degree=2)
    assert design.shape == (len(x), 3)
    assert np.allclose(design[:, 0], np.ones(len(x)))


def test_polynomial_training_and_prediction():
    x, y = load_regression_1d()
    weights = regression.fit_polynomial_regression(x, y, degree=2)
    assert weights.shape == (3,)
    expected = load_expected_weights("expected_polynomial_weights.csv")
    assert np.allclose(weights, expected, atol=0.05)
    preds = regression.predict_polynomial(x, weights)
    mse = np.mean((preds - y) ** 2)
    assert mse < 0.05


def test_surface_training_and_prediction():
    x1, x2, y = load_regression_2d()
    weights = regression.fit_surface_regression(x1, x2, y)
    assert weights.shape == (6,)
    preds = regression.predict_surface(x1, x2, weights)
    mse = np.mean((preds - y) ** 2)
    assert mse < 0.02
    expected = load_expected_weights("expected_surface_weights.csv")
    assert np.allclose(weights, expected, atol=0.08)

