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
    
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z: np.ndarray) -> np.ndarray:
    """
    Apply a numerically stable softmax across rows.

    Args:
        z (np.ndarray): Logit matrix of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Probabilities for each class per sample.
    """
    max_z = np.max(z, axis=1, keepdims=True)
    
    exp_z = np.exp(z - max_z)

    sum_exp = np.sum(exp_z, axis=1, keepdims=True)

    return exp_z / sum_exp


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

        
        
        X_arr = np.array(X)
        y_arr = np.array(y)

        if X_arr.shape[0] == 0:
            return # Nothing to learn
        
        if X_arr.ndim != 2:
             # Fix: Reshape 1D input to 2D column vector if needed, or raise error
             # Standard sklearn behavior expects 2D, so we raise error if strictly 1D
             raise ValueError(f"Input must be 2D, but got shape {X_arr.shape}")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        # Check for valid binary labels
        unique = np.unique(y_arr)
        if not np.all(np.isin(unique, [0, 1])):
             raise ValueError("Labels must be binary (0 or 1).")

        n_features = X_arr.shape[1]
        self._initialize_parameters(n_features)

        # TRAINING LOOP
        for _ in range(self.epochs):
            z, probs = self._forward(X_arr)
            grad_w, grad_b = self._backward(X_arr, y_arr, probs)
            self._update(grad_w, grad_b)


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
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Inside predict_proba (both classes)
        X_arr = np.array(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1) # Treat as 1 sample with N features
                    
        _, probs = self._forward(X_arr)
        return probs

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels (0 or 1) using a 0.5 threshold.
        """
        probs = self.predict_proba(X)
        # Convert probabilities to 0 or 1. Result is flat array.
        return (probs >= 0.5).astype(int).flatten()

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and probabilities for the current parameters.
        """
        z = np.dot(X,self.weights) + self.bias
        p = _sigmoid(z)

        return z,p

    def _backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients of the loss with respect to weights and bias.
        """
        n = X.shape[0]

        
        # Ensure y is (n, 1) to match probs shape
        # If probs is (n, 1) and y is (n,), subtraction broadcasts wrongly
        if probs.ndim == 1:
             probs = probs.reshape(-1, 1)
        
        y_true = y_true.reshape(probs.shape)
        
        e = probs - y_true
        
        # Gradient w: (1/n) * X.T @ e + lambda * w
        # CRITICAL: Use @ (matrix mult), not * (element-wise)
        grad_w = (1 / n) * (X.T @ e).flatten() + (self.reg_strength * self.weights)
        
        # Gradient b: (1/n) * sum(e)
        grad_b = (1 / n) * np.sum(e)
        
        return grad_w, grad_b


    def _update(self, grad_w: np.ndarray, grad_b: float) -> None:
        """
        Apply one gradient descent step.
        """
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialise weights from a small Gaussian and zero bias.
        """

        self.bias = 0.0
        self.weights = self._rng.normal(loc=0.0, scale=0.01, size=(n_features,))


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
        X_arr = np.array(X)
        y_arr = np.array(y)

        if X_arr.shape[0] == 0:
            return # Nothing to learn
        
        # 1. Identify classes
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        n_features = X_arr.shape[1]
        n_samples = X_arr.shape[0]

        # 2. One-hot encode targets
        # Create a map from label -> index (e.g., 'cat'->0, 'dog'->1)
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        
        # Build Y matrix (n_samples, n_classes)
        y_indices = np.array([label_to_idx[label] for label in y_arr])
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y_indices] = 1.0

        # 3. Initialize
        self._initialize_parameters(n_features, n_classes)

        # 4. Training Loop
        for _ in range(self.epochs):
            _, probs = self._forward(X_arr)
            grad_w, grad_b = self._backward(X_arr, Y_onehot, probs)
            self._update(grad_w, grad_b)

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
        
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
            
        X_arr = np.array(X)
        _, probs = self._forward(X_arr)
        return probs

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels via argmax over predicted probabilities.
        """
        probs = self.predict_proba(X)
        # Find index of max probability for each row
        max_indices = np.argmax(probs, axis=1)
        # Map indices back to original class labels
        return self.classes_[max_indices]

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute logits and softmax probabilities for the current parameters.
        """
        z = X @ self.weights + self.bias
        p = _softmax(z)
        return z, p

    def _backward(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and bias given softmax probabilities.
        """
        n = X.shape[0]

        
        # Error = P - Y
        e = probs - y_onehot
        
        # Grad W = (1/n) * X.T @ E + reg * W
        grad_w = (1 / n) * (X.T @ e) + (self.reg_strength * self.weights)
        
        # Grad b = (1/n) * sum(E, axis=0) -> Sum down rows to get one bias per class
        grad_b = (1 / n) * np.sum(e, axis=0)
        
        return grad_w, grad_b

    def _update(self, grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        """
        Apply one gradient descent update.
        """
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def _initialize_parameters(self, n_features: int, n_classes: int) -> None:
        """
        Initialise weights and biases for a given feature/class configuration.
        """
        self.weights = self._rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
        # Bias: (classes,)
        self.bias = np.zeros(n_classes)

