import sys
from pathlib import Path

import pandas as pd
import pytest

# Add code directory to path (following bayesnet pattern)
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import decision_tree


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


def load_dataset():
    return pd.read_csv(DATA_DIR / "decision_tree.csv")


def load_expected_metrics():
    return pd.read_csv(EXPECTED_DIR / "expected_decision_tree_metrics.csv")


def load_expected_base():
    df = pd.read_csv(EXPECTED_DIR / "expected_decision_tree_base.csv")
    return {row["metric"]: float(row["value"]) for _, row in df.iterrows()}


def load_expected_best():
    df = pd.read_csv(EXPECTED_DIR / "expected_decision_tree_best.csv")
    return {row["criterion"]: row["feature"] for _, row in df.iterrows()}


def test_entropy_and_gini_dataset():
    df = load_dataset()
    labels = df["play"].tolist()
    expected = load_expected_base()
    assert decision_tree.entropy(labels) == pytest.approx(expected["entropy"], rel=1e-9)
    assert decision_tree.gini(labels) == pytest.approx(expected["gini"], rel=1e-9)


def test_information_gain_and_gain_ratio():
    df = load_dataset()
    expected_metrics = load_expected_metrics()
    for _, row in expected_metrics.iterrows():
        feature = row["feature"]
        assert decision_tree.information_gain(df, feature) == pytest.approx(
            row["information_gain"], rel=1e-9
        )
        assert decision_tree.gain_ratio(df, feature) == pytest.approx(row["gain_ratio"], rel=1e-9)


def test_best_split_by_criteria():
    df = load_dataset()
    features = [col for col in df.columns if col not in {"day", "play"}]
    expected = load_expected_best()
    for criterion, feature in expected.items():
        assert (
            decision_tree.best_split(df, features, target="play", criterion=criterion) == feature
        )

