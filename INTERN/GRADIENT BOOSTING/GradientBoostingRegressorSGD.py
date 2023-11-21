import numpy as np
from typing import Tuple
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss="mse",
            verbose=False,
            subsample_size=0.5,
            replace=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        self.base_pred_ = None
        self.subsample_size = subsample_size
        self.replace = replace

    def _subsample(self, X, y):
        size = int(len(X) * self.subsample_size)
        idx = np.random.choice(np.arange(0,len(X)), size=size, replace=self.replace)
        sub_x = X[idx,:]
        sub_y = y[idx]
        return sub_x, sub_y

    def _mae(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        """Mean absolute error loss function and gradient."""
        loss = np.sum(np.abs(y_pred - y_true)) / len(y_true)
        grad = np.sign(y_pred - y_true)
        return loss, grad

    def _mse(self, y_true, y_pred) -> Tuple[float, np.ndarray]:
        """Mean square error loss function and gradient."""
        loss = np.sum(np.square(y_pred-y_true)) / len(y_true)
        grad = y_pred - y_true
        return loss, grad

    def _huber(self, y_true, y_pred, delta=1.0):
        # Huber Loss функция потерь
        residual = np.abs(y_true - y_pred)
        loss = np.where(residual <= delta, 0.5 * residual**2, delta * residual - 0.5 * delta**2)
        return loss

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = np.mean(y).astype(np.float64)

        y_pred = np.full_like(y, self.base_pred_, dtype=np.float64)
        iteration = 0
        while len(self.trees_) < self.n_estimators:

            if self.loss == "mse":
                loss, pseudo_residuals = self._mse(y, y_pred)
            elif self.loss == "mae":
                loss, pseudo_residuals = self._mae(y, y_pred)
            elif self.loss == "huber":
                pseudo_residuals = self._huber(y, y_pred)
            else:
                loss, pseudo_residuals = self.loss(y, y_pred)

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split)
            antigrad = -pseudo_residuals
            tree.fit(*self._subsample(X, antigrad))

            self.trees_.append(tree)

            y_pred += self.learning_rate * np.array(tree.predict(X)).astype(np.float64)
            iteration += 1
            if self.verbose:
                loss, grad = self._mse(y, y_pred)
                print(f"Iteration {iteration}, MSE: {loss}")

        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        y_pred = np.full(X.shape[0], self.base_pred_)

        for tree in self.trees_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
