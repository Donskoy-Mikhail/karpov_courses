from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mean squared error of a vector."""
        me = np.mean(y)
        res = np.mean((y - me) ** 2)
        return res

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mean squared error of two vectors."""
        res = ((self._mse(y_left) * len(y_left) + self._mse(y_right) * len(y_right))
               / (len(y_left) + len(y_right)))
        return res

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for all features"""

        def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
            """Find the best split for a node (one feature)"""

            unique_values = np.unique(X[:, feature])

            best_threshold = None
            best_weighted_mse = float('inf')

            for threshold in unique_values:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
                right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)

                weighted_mse = (left_mse * np.sum(left_indices) + right_mse * np.sum(right_indices)) / len(y)

                if weighted_mse < best_weighted_mse:
                    best_weighted_mse = weighted_mse
                    best_threshold = threshold

            return best_threshold

        best_idx = None
        best_thr = None
        best_weighted_mse = float('inf')

        for feature in range(X.shape[1]):
            threshold = split(X, y, feature)

            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold

            left_mse = self._mse(y[left_indices])
            right_mse = self._mse(y[right_indices])

            weighted_mse = (left_mse * np.sum(left_indices) + right_mse * np.sum(right_indices)) / len(y)

            if weighted_mse < best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_idx = feature
                best_thr = threshold

        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        mse = self._mse(y)
        if depth == self.max_depth or len(y) < self.min_samples_split:
            return Node(value=round(float(np.mean(y))), n_samples=len(y), mse=mse)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return Node(value=round(float(np.mean(y))), n_samples=len(y), mse=mse)
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_node = self._split_node(X[left_indices], y[left_indices], depth + 1)
        right_node = self._split_node(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, n_samples=len(y),
                    value=round(np.mean(y)), mse=mse, left=left_node, right=right_node)

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return self._as_json(self.tree_)

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.feature is None:
            # Если это лист, формируем JSON для листа
            return f'{{"value": {node.value}, "n_samples": {node.n_samples}, "mse": {round(node.mse,2)}}}'
        else:
            # Если это узел, формируем JSON для узла и рекурсивно для его потомков (left и right)
            left_json = self._as_json(node.left)
            right_json = self._as_json(node.right)
            return (f'{{"feature": {node.feature}, "threshold":'
                    f' {node.threshold}, "n_samples": {node.n_samples},'
                    f' "mse": {round(node.mse,2)}, "left": {left_json}, "right": {right_json}}}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        predictions = []
        for sample in X:
            prediction = self._predict_one_sample(sample)
            predictions.append(prediction)
        return np.array(predictions)

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        node = self.tree_
        while node.left is not None and node.right is not None:
            if features[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value