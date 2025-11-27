import sys
from pathlib import Path

import pandas as pd
import pytest

# Add code directory to path (following bayesnet pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import metrics_classification
import metrics_regression


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


def load_classification():
    df = pd.read_csv(DATA_DIR / "metrics_classification.csv")
    return df["y_true"].tolist(), df["y_pred"].tolist()


def load_regression():
    df = pd.read_csv(DATA_DIR / "metrics_regression.csv")
    return df["y_true"].tolist(), df["y_pred"].tolist()


def load_expected_metrics(filename: str) -> dict:
    df = pd.read_csv(EXPECTED_DIR / filename)
    return {row["metric"]: float(row["value"]) for _, row in df.iterrows()}


def test_classification_metrics_cat_label():
    y_true, y_pred = load_classification()
    expected = load_expected_metrics("expected_metrics_classification.csv")
    assert metrics_classification.accuracy_score(y_true, y_pred) == pytest.approx(expected["accuracy"])
    assert metrics_classification.precision_score(y_true, y_pred, positive_label="cat") == pytest.approx(
        expected["precision_cat"]
    )
    assert metrics_classification.recall_score(y_true, y_pred, positive_label="cat") == pytest.approx(
        expected["recall_cat"]
    )
    assert metrics_classification.f1_score(y_true, y_pred, positive_label="cat") == pytest.approx(
        expected["f1_cat"], rel=1e-9
    )


def test_macro_micro_scores():
    y_true, y_pred = load_classification()
    expected = load_expected_metrics("expected_metrics_classification.csv")
    labels = ["cat", "dog", "bird"]
    assert metrics_classification.macro_f1_score(y_true, y_pred, labels=labels) == pytest.approx(
        expected["macro_f1"], rel=1e-6
    )
    assert metrics_classification.micro_f1_score(y_true, y_pred, labels=labels) == pytest.approx(
        expected["micro_f1"], rel=1e-9
    )


def test_regression_metrics():
    y_true, y_pred = load_regression()
    expected = load_expected_metrics("expected_metrics_regression.csv")
    assert metrics_regression.mean_absolute_error(y_true, y_pred) == pytest.approx(expected["mae"], rel=1e-6)
    assert metrics_regression.mean_squared_error(y_true, y_pred) == pytest.approx(expected["mse"], rel=1e-6)
    assert metrics_regression.root_mean_squared_error(y_true, y_pred) == pytest.approx(
        expected["rmse"], rel=1e-6
    )
    assert metrics_regression.r2_score(y_true, y_pred) == pytest.approx(expected["r2"], rel=1e-6)


def test_regression_report_structure():
    y_true, y_pred = load_regression()
    expected = load_expected_metrics("expected_metrics_regression.csv")
    report = metrics_regression.regression_report(y_true, y_pred)
    assert set(report.keys()) == {"mae", "mse", "rmse", "r2"}
    assert report["mae"] == pytest.approx(expected["mae"], rel=1e-6)
    assert report["r2"] == pytest.approx(expected["r2"], rel=1e-6)
