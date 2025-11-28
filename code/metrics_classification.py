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
    if denom == 0:
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
    y_true_arr = np.array(y_true).ravel()
    y_pred_arr = np.array(y_pred).ravel()

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("Arrays do not share the same length")
    
    return y_true_arr, y_pred_arr


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
    if labels is None:
        # Concatenate and find unique values (sorted automatically by unique)
        return np.unique(np.concatenate((y_true_arr, y_pred_arr)))
    
    # If labels are provided, return them as a numpy array
    return np.array(labels)



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
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)

    resolved_labels = _resolve_labels(labels, y_true_arr, y_pred_arr)

    label= {label: i for i, label in enumerate(resolved_labels)}

    n_labels = len(resolved_labels)
    confusion_matrix1 = np.zeros((n_labels, n_labels), dtype=int)

    for true_val, pred_val in zip(y_true_arr, y_pred_arr):
        row_idx = label.get(true_val)
        col_idx = label.get(pred_val)
        if row_idx is not None and col_idx is not None:
            confusion_matrix1[row_idx, col_idx] += 1
            
    return confusion_matrix1


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
    matrix = confusion_matrix(y_true,y_pred,None)
    correct = np.trace(matrix)

    total = matrix.sum()

    return _safe_divide(correct, total)


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

    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    
    labels_list = _resolve_labels(None, y_true_arr, y_pred_arr)
    
    try:
        search_list = labels_list.tolist() if isinstance(labels_list, np.ndarray) else labels_list
        index = search_list.index(positive_label)
    except ValueError:
        return 0.0

    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_list)
    true_positive = matrix[index, index]
    total_predicted_positive = matrix[:, index].sum()

    return _safe_divide(true_positive, total_predicted_positive)

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
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    labels_list = _resolve_labels(None, y_true_arr, y_pred_arr)
    
    try:
        search_list = labels_list.tolist() if isinstance(labels_list, np.ndarray) else labels_list
        index = search_list.index(positive_label)
    except ValueError:
        return 0.0

    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_list)
    
    true_positive = matrix[index, index]
    total_actual_positive = matrix[index, :].sum()

    return _safe_divide(true_positive, total_actual_positive)

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
    p = precision_score(y_true, y_pred, positive_label)
    r = recall_score(y_true, y_pred, positive_label)
    

    return _safe_divide(2 * p * r, p + r)


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
    scores = []
    for label in labels:
        score = f1_score(y_true, y_pred, positive_label=label)
        scores.append(score)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


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
    y_true_arr, y_pred_arr = _prepare_inputs(y_true, y_pred)
    
    # We need the full set of unique labels in the data to build the matrix
    all_labels = _resolve_labels(None, y_true_arr, y_pred_arr)
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=all_labels)
    
    # Map labels to indices for quick lookup
    label_to_index = {lbl: i for i, lbl in enumerate(all_labels)}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # [cite_start]2. Aggregate counts for the requested labels [cite: 173]
    for label in labels:
        if label in label_to_index:
            idx = label_to_index[label]
            
            # TP: Diagonal value
            tp = matrix[idx, idx]
            
            # FP: Column sum - TP
            fp = matrix[:, idx].sum() - tp
            
            # FN: Row sum - TP
            fn = matrix[idx, :].sum() - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

    # 3. Compute Micro Metrics
    # Micro Precision = Sum(TP) / (Sum(TP) + Sum(FP))
    micro_precision = _safe_divide(total_tp, total_tp + total_fp)
    
    # Micro Recall = Sum(TP) / (Sum(TP) + Sum(FN))
    micro_recall = _safe_divide(total_tp, total_tp + total_fn)
    
    # [cite_start]4. Return Harmonic Mean [cite: 175]
    return _safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

