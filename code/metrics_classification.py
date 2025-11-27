"""Reference implementation for classification metrics (Part 1A)."""

from __future__ import annotations

import numpy as np


def _safe_divide(num: float, denom: float) -> float:
    """
    Divide two floats safely, returning 0.0 when the denominator is zero.

    Args:
        num (float): Numerator.
        denom (float): Denominator.

    Returns:
        float: num / denom when denom != 0, otherwise 0.0.
    """
    if demo == 0:
        return 0.0
    return num / denom


def _prepare_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classification targets/predictions into 1-D NumPy arrays.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pair of flattened arrays (y_true, y_pred).

    Raises:
        ValueError: If the arrays do not share the same length.
    """

    
    raise NotImplementedError("Implement _prepare_inputs.")


def _resolve_labels(labels, y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> list:
    """
    Determine the ordered label set used to build the confusion matrix.

    Args:
        labels (array-like | None): Explicit label ordering or None to infer.
        y_true_arr (np.ndarray): Flattened true labels.
        y_pred_arr (np.ndarray): Flattened predicted labels.

    Returns:
        list: Ordered list of unique labels.

    Raises:
        ValueError: If the final label list is empty.
    """
    raise NotImplementedError("Implement _resolve_labels.")


def confusion_matrix(
    y_true,
    y_pred,
    labels=None,
) -> np.ndarray:
    """
    Build the confusion matrix for multi-class classification.

    Args:
        y_true (array-like): True labels, convertible to a 1-D NumPy array.
        y_pred (array-like): Predicted labels, same length as `y_true`.
        labels (array-like | None): Optional ordered list of label values. When
            None, the union of labels from y_true and y_pred (in encounter order)
            is used. Every element must be hashable.

    Returns:
        np.ndarray: Square matrix of shape (n_classes, n_classes) containing
            integer counts. Rows correspond to true labels and columns to
            predicted labels.

    Example:
        >>> confusion_matrix(["cat", "dog"], ["cat", "cat"], labels=["cat", "dog"])
        array([[1, 0],
               [1, 0]])
    """
    raise NotImplementedError("Implement confusion_matrix.")


def accuracy_score(y_true, y_pred) -> float:
    """
    Compute overall accuracy from the confusion matrix.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Ratio of correctly predicted samples over total samples.

    Example:
        >>> accuracy_score(["cat", "dog"], ["cat", "cat"])
        0.5
    """
    raise NotImplementedError("Implement accuracy_score.")


def precision_score(y_true, y_pred, positive_label) -> float:
    """
    Precision for a single positive class, computed from the confusion matrix.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        positive_label: Label treated as the positive class.

    Returns:
        float: True positives divided by predicted positives.

    Example:
        >>> precision_score(["cat", "dog"], ["cat", "cat"], positive_label="cat")
        0.5
    """
    raise NotImplementedError("Implement precision_score.")


def recall_score(y_true, y_pred, positive_label) -> float:
    """
    Recall for a single positive class, computed from the confusion matrix.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        positive_label: Label treated as the positive class.

    Returns:
        float: True positives divided by actual positives (support).

    Example:
        >>> recall_score(["cat", "cat"], ["cat", "dog"], positive_label="cat")
        0.5
    """
    raise NotImplementedError("Implement recall_score.")


def f1_score(y_true, y_pred, positive_label) -> float:
    """
    Harmonic mean of precision and recall for a single positive class.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        positive_label: Label treated as the positive class.

    Returns:
        float: 2 * precision * recall / (precision + recall).

    Example:
        >>> f1_score(["cat", "dog"], ["cat", "cat"], positive_label="cat")
        0.6666666666666666
    """
    raise NotImplementedError("Implement f1_score.")


def macro_f1_score(y_true, y_pred, labels) -> float:
    """
    Average F1 score across all specified classes (unweighted macro average).

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        labels (array-like): Ordered list of labels to include in the macro
            computation.

    Returns:
        float: Mean of per-class F1 scores.

    Example:
        >>> macro_f1_score(["cat", "dog"], ["cat", "cat"], labels=["cat", "dog"])
        0.5
    """
    raise NotImplementedError("Implement macro_f1_score.")


def micro_f1_score(y_true, y_pred, labels) -> float:
    """
    Micro-averaged F1 score aggregated over all specified classes.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        labels (array-like): Label set to include when computing micro F1.

    Returns:
        float: F1 score derived from global TP/FP/FN sums.

    Example:
        >>> micro_f1_score(["cat", "dog"], ["cat", "cat"], labels=["cat", "dog"])
        0.5
    """
    raise NotImplementedError("Implement micro_f1_score.")

