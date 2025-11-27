"""Reference implementations for Part 3 – logistic and softmax regression."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Apply the logistic sigmoid element-wise.

    Args:
        z (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid outputs.
    """
    raise NotImplementedError("Implement _sigmoid.")


def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply a numerically stable softmax across rows.

    Args:
        z (np.ndarray): Logit matrix of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Probabilities for each class per sample.
    """
    raise NotImplementedError("Implement _softmax.")


class LogisticRegression:
    """Binary logistic regression trained via batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1500,
        reg_strength: float = 0.0,
        random_state: int | None = 0,
    ) -> None:
        """
        Args:
            learning_rate (float): Step size for gradient updates (> 0).
            epochs (int): Number of passes over the training data (> 0).
            reg_strength (float): L2 regularisation strength (>= 0).
            random_state (int | None): Seed passed to NumPy default RNG.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if reg_strength < 0:
            raise ValueError("reg_strength cannot be negative.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y) -> None:
        """
        Train the classifier using batch gradient descent.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Binary labels of shape (n_samples,).
        """
        raise NotImplementedError("Implement LogisticRegression.fit.")

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for each sample.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Probabilities for the positive class.

        Raises:
            RuntimeError: If called before `fit`.
        """
        raise NotImplementedError("Implement LogisticRegression.predict_proba.")

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels (0 or 1) using a 0.5 threshold.
        """
        raise NotImplementedError("Implement LogisticRegression.predict.")

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and probabilities for the current parameters.
        """
        raise NotImplementedError("Implement LogisticRegression._forward.")

    def _backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients of the loss with respect to weights and bias.
        """
        raise NotImplementedError("Implement LogisticRegression._backward.")

    def _update(self, grad_w: np.ndarray, grad_b: float) -> None:
        """
        Apply one gradient descent step.
        """
        raise NotImplementedError("Implement LogisticRegression._update.")

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialise weights from a small Gaussian and zero bias.
        """
        raise NotImplementedError("Implement LogisticRegression._initialize_parameters.")


class SoftmaxRegression:
    """Multiclass generalisation of logistic regression with softmax output."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 2000,
        reg_strength: float = 0.0,
        random_state: int | None = 0,
    ) -> None:
        """
        Args:
            learning_rate (float): Step size for gradient updates (> 0).
            epochs (int): Number of iterations (> 0).
            reg_strength (float): L2 penalty applied to weights (>= 0).
            random_state (int | None): Seed for reproducible initialisation.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if reg_strength < 0:
            raise ValueError("reg_strength cannot be negative.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y) -> None:
        """
        Train the model using gradient descent on the cross-entropy loss.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Class labels (hashable) of shape (n_samples,).
        """
        raise NotImplementedError("Implement SoftmaxRegression.fit.")

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for each sample.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Shape (n_samples, n_classes) with row sums equal to 1.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        raise NotImplementedError("Implement SoftmaxRegression.predict_proba.")

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels via argmax over predicted probabilities.
        """
        raise NotImplementedError("Implement SoftmaxRegression.predict.")

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and softmax probabilities for the current parameters.
        """
        raise NotImplementedError("Implement SoftmaxRegression._forward.")

    def _backward(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and bias given softmax probabilities.
        """
        raise NotImplementedError("Implement SoftmaxRegression._backward.")

    def _update(self, grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        """
        Apply one gradient descent update.
        """
        raise NotImplementedError("Implement SoftmaxRegression._update.")

    def _initialize_parameters(self, n_features: int, n_classes: int) -> None:
        """
        Initialise weights and biases for a given feature/class configuration.
        """
        raise NotImplementedError("Implement SoftmaxRegression._initialize_parameters.")

