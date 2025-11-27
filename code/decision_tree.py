"""Reference utilities for decision tree metrics and best-split selection."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import math
import pandas as pd


def entropy(labels: Sequence[str]) -> float:
    """
    Compute Shannon entropy (base 2) from a multiset of labels.

    Args:
        labels (Sequence[str]): Iterable of categorical labels.

    Returns:
        float: Entropy value in bits. Returns 0.0 for empty input.
    """
    raise NotImplementedError("Implement entropy.")


def gini(labels: Sequence[str]) -> float:
    """
    Compute the Gini impurity for the provided labels.

    Args:
        labels (Sequence[str]): Iterable of categorical labels.

    Returns:
        float: Gini impurity (0.0 indicates pure set).
    """
    raise NotImplementedError("Implement gini.")


def partition_dataset(df: pd.DataFrame, feature: str) -> Dict[str, pd.DataFrame]:
    """
    Group rows of a dataframe by a categorical feature.

    Args:
        df (pd.DataFrame): Input dataset.
        feature (str): Column name to partition on.

    Returns:
        Dict[str, pd.DataFrame]: Mapping from feature value to subset dataframe
        (reindexed from 0).
    """
    raise NotImplementedError("Implement partition_dataset.")


def information_gain(
    df: pd.DataFrame,
    feature: str,
    target: str = "play",
) -> float:
    """
    Compute information gain of splitting on a categorical feature.

    Args:
        df (pd.DataFrame): Dataset containing feature and target columns.
        feature (str): Feature to evaluate.
        target (str): Target column name (default "play").

    Returns:
        float: Information gain in bits.
    """
    raise NotImplementedError("Implement information_gain.")


def _split_info(partitions: Dict[str, pd.DataFrame], total_rows: int) -> float:
    """
    Compute the split information term used in gain ratio.

    Args:
        partitions (Dict[str, pd.DataFrame]): Subsets after splitting.
        total_rows (int): Total number of rows in the original dataset.

    Returns:
        float: Split information (entropy of partition proportions).
    """
    raise NotImplementedError("Implement _split_info.")


def gain_ratio(
    df: pd.DataFrame,
    feature: str,
    target: str = "play",
) -> float:
    """
    Compute the gain ratio of splitting on `feature`.

    Args:
        df (pd.DataFrame): Dataset containing feature and target columns.
        feature (str): Feature to evaluate.
        target (str): Target column name.

    Returns:
        float: Gain ratio (0 when split information is zero).
    """
    raise NotImplementedError("Implement gain_ratio.")


def best_split(
    df: pd.DataFrame,
    candidate_features: Iterable[str],
    target: str = "play",
    criterion: str = "gain_ratio",
) -> str:
    """
    Select the best feature to split on using the specified criterion.

    Args:
        df (pd.DataFrame): Dataset including candidate feature columns.
        candidate_features (Iterable[str]): Feature names to evaluate.
        target (str): Target column name.
        criterion (str): One of {"gain_ratio", "information_gain", "gini"}.

    Returns:
        str: Feature name with highest score.

    Raises:
        ValueError: If an invalid criterion is provided or no candidates exist.
    """
    raise NotImplementedError("Implement best_split.")

