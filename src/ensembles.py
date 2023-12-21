import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        if feature_subsample_size is None:
            feature_subsample_size = 1 / 3
        self.feature_subsample_size = feature_subsample_size
        self.trees = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for _ in range(n_estimators)]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        if X_val is not None and y_val is not None:
            rmse = []

        self.features_num = []
        block_size = int(X.shape[0] * (1 - 1 / np.e))
        feature_size = int(X.shape[1] * self.feature_subsample_size)
        for i in range(self.n_estimators):
            objects = np.random.choice(X.shape[0], block_size, replace=False)
            features = np.random.choice(X.shape[1], feature_size, replace=False)
            self.features_num.append(features)
            self.trees[i].fit(X[objects][:, features], y[objects])

            if X_val is not None and y_val is not None:
                rmse.append(mean_squared_error(y_val, self.predict(X_val), squared=False))
        
        if X_val is not None and y_val is not None:
            return np.array(rmse)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        answers = []
        for i in range(self.n_estimators):
            answers.append(self.trees[i].predict(X[:, self.features_num[i]]))

        return np.mean(np.array(answers), axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate 
        self.max_depth = max_depth
        if feature_subsample_size is None:
            feature_subsample_size = 1 / 3
        self.feature_subsample_size = feature_subsample_size
        self.trees = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for _ in range(n_estimators)]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        if X_val is not None and y_val is not None:
            rmse = []

        self.features_num = []
        self.coefs= []
        block_size = int(X.shape[0] * (1 - 1 / np.e))
        feature_size = int(X.shape[1] * self.feature_subsample_size)
        a = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            objects = np.random.choice(X.shape[0], block_size, replace=False)
            features = np.random.choice(X.shape[1], feature_size, replace=False)
            self.features_num.append(features)
            grad = 2 * (y - a)
            self.trees[i].fit(X[objects][:, features], grad[objects])
            y_pred = self.trees[i].predict(X[:, features])
            loss = lambda alpha: np.mean(((a + alpha * self.learning_rate * y_pred) - y) ** 2)
            best_alpha = minimize_scalar(loss)
            self.coefs.append(best_alpha.x * self.learning_rate)
            a += self.coefs[-1] * y_pred

            if X_val is not None and y_val is not None:
                rmse.append(mean_squared_error(y_val, self.predict(X_val), squared=False))
        
        if X_val is not None and y_val is not None:
            return np.array(rmse)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        answers = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            answers += self.coefs[i] * self.trees[i].predict(X[:, self.features_num[i]])

        return answers
