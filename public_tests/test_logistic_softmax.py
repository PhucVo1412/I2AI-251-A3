import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add code directory to path (following bayesnet pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from logistic_softmax import LogisticRegression, SoftmaxRegression


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


def load_binary():
    df = pd.read_csv(DATA_DIR / "logistic_binary.csv")
    X = df[["x1", "x2"]].to_numpy()
    y = df["label"].to_numpy()
    return X, y


def load_multiclass():
    df = pd.read_csv(DATA_DIR / "softmax_multiclass.csv")
    X = df[["x1", "x2"]].to_numpy()
    y = df["label"].to_numpy()
    return X, y


def load_expected_logistic():
    df = pd.read_csv(EXPECTED_DIR / "expected_logistic.csv").sort_values("sample_index")
    return df["probability"].to_numpy(), df["pred_label"].to_numpy()


def load_expected_softmax():
    df = pd.read_csv(EXPECTED_DIR / "expected_softmax.csv").sort_values("sample_index")
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    probs = df[prob_cols].to_numpy()
    preds = df["pred_label"].to_numpy()
    classes = [col.replace("prob_", "") for col in prob_cols]
    return probs, preds, classes


def test_logistic_regression_trains_and_predicts():
    X, y = load_binary()
    clf = LogisticRegression(learning_rate=0.3, epochs=4000, reg_strength=0.01)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    expected_probs, expected_labels = load_expected_logistic()
    assert probs.shape == (len(X),)
    assert np.allclose(probs, expected_probs, atol=1e-4)
    preds = clf.predict(X)
    assert preds.tolist() == expected_labels.tolist()


def test_softmax_regression_trains_and_predicts():
    X, y = load_multiclass()
    clf = SoftmaxRegression(learning_rate=0.2, epochs=5000, reg_strength=0.01)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    expected_probs, expected_labels, classes = load_expected_softmax()
    assert probs.shape == expected_probs.shape
    assert np.allclose(np.sum(probs, axis=1), np.ones(len(X)), atol=1e-6)
    assert np.allclose(probs, expected_probs, atol=1e-4)
    preds = clf.predict(X)
    assert preds.tolist() == expected_labels.tolist()

