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
    n = len(labels)
    if n == 0:
        return 0.0
    
    # Count frequency of each label
    counts = Counter(labels)
    
    ent = 0.0
    for count in counts.values():
        p = count / n
        # Formula: - sum(p * log2(p))
        if p > 0:
            ent -= p * math.log2(p)
            
    return ent


def gini(labels: Sequence[str]) -> float:
    """
    Compute the Gini impurity for the provided labels.

    Args:
        labels (Sequence[str]): Iterable of categorical labels.

    Returns:
        float: Gini impurity (0.0 indicates pure set).
    """
    n = len(labels)
    if n == 0:
        return 0.0
    
    counts = Counter(labels)
    
    # Formula: 1 - sum(p^2)
    sum_sq_p = 0.0
    for count in counts.values():
        p = count / n
        sum_sq_p += p**2
        
    return 1.0 - sum_sq_p


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
    partitions = {}
    
    # Use pandas groupby to split the dataframe
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

    for val, group in df.groupby(feature):
        # Reset index is required by spec/docstring
        partitions[val] = group.reset_index(drop=True)
        
    return partitions


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
    parent_entropy = entropy(df[target])
    
    # 2. Partition S into subsets {Sv}
    partitions = partition_dataset(df, feature)
    n_total = len(df)
    
    # 3. Calculate Weighted Average Entropy of children
    weighted_child_entropy = 0.0
    for subset in partitions.values():
        weight = len(subset) / n_total
        weighted_child_entropy += weight * entropy(subset[target])
        
    # 4. IG = H(S) - WeightedChildEntropy
    return parent_entropy - weighted_child_entropy



def _split_info(partitions: Dict[str, pd.DataFrame], total_rows: int) -> float:
    """
    Compute the split information term used in gain ratio.

    Args:
        partitions (Dict[str, pd.DataFrame]): Subsets after splitting.
        total_rows (int): Total number of rows in the original dataset.

    Returns:
        float: Split information (entropy of partition proportions).
    """
    split_info_val = 0.0
    
    for subset in partitions.values():
        p = len(subset) / total_rows
        if p > 0:
            split_info_val -= p * math.log2(p)
            
    return split_info_val


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
    ig = information_gain(df, feature, target)
    
    partitions = partition_dataset(df, feature)
    si = _split_info(partitions, len(df))
    
    # Handle division by zero (if SplitInfo is 0, usually means 1 partition -> useless split)
    if si == 0:
        return 0.0
        
    return ig / si

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
    candidates = list(candidate_features)
    if not candidates:
        raise ValueError("Candidate features list is empty.")
    
    best_feature = None
    best_score = -float('inf')
    
    for feature in candidates:
        score = 0.0
        
        if criterion == "information_gain":
            score = information_gain(df, feature, target)
            
        elif criterion == "gain_ratio":
            score = gain_ratio(df, feature, target)
            
        elif criterion == "gini":
            # Gini Gain = Parent Gini - Weighted Child Gini
            parent_gini = gini(df[target])
            partitions = partition_dataset(df, feature)
            n_total = len(df)
            
            weighted_child_gini = 0.0
            for subset in partitions.values():
                weight = len(subset) / n_total
                weighted_child_gini += weight * gini(subset[target])
                
            score = parent_gini - weighted_child_gini
            
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
        # Update best feature if this score is better
        # Note: For all criteria implemented here, "higher is better"
        if score > best_score:
            best_score = score
            best_feature = feature
            
    return best_feature

