import numpy as np


class DecisionTreeNode:
    """
    Decision tree node
    """
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False


class DecisionTreeClassifier:
    """
    Decision Tree Classifier for multi-class classification

    Splitting criterion: Gini impurity
    Gini(D) = 1 - sum(pk^2), where pk is proportion of class k

    Information gain:
    Gain(D, A) = Gini(D) - sum((|Dv|/|D|) * Gini(Dv))
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize decision tree classifier

        Args:
            max_depth: maximum depth of tree
            min_samples_split: minimum samples required to split node
            min_samples_leaf: minimum samples required at leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.classes_ = None
        self.n_classes_ = None

    def gini_impurity(self, y):
        """
        Calculate Gini impurity

        Gini = 1 - sum(pk^2)

        Args:
            y: labels

        Returns:
            gini: Gini impurity value
        """
        if len(y) == 0:
            return 0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)

        return gini

    def split_dataset(self, X, y, feature_idx, threshold):
        """
        Split dataset by feature and threshold

        Args:
            X: features
            y: labels
            feature_idx: index of feature to split on
            threshold: threshold value

        Returns:
            left_X, left_y, right_X, right_y: split datasets
        """
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        return left_X, left_y, right_X, right_y

    def find_best_split(self, X, y):
        """
        Find best feature and threshold to split on

        Args:
            X: features, shape (n_samples, n_features)
            y: labels, shape (n_samples,)

        Returns:
            best_feature_idx: index of best feature
            best_threshold: best threshold value
            best_gain: best information gain
        """
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        current_gini = self.gini_impurity(y)

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_X, left_y, right_X, right_y = self.split_dataset(X, y, feature_idx, threshold)

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                left_gini = self.gini_impurity(left_y)
                right_gini = self.gini_impurity(right_y)

                n_left, n_right = len(left_y), len(right_y)
                weighted_gini = (n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini

                gain = current_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        """
        Recursively build decision tree

        Args:
            X: features
            y: labels
            depth: current depth

        Returns:
            node: tree node
        """
        n_samples = len(y)

        node = DecisionTreeNode()

        if depth == self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            node.is_leaf = True
            unique, counts = np.unique(y, return_counts=True)
            node.value = unique[np.argmax(counts)]
            return node

        feature_idx, threshold, gain = self.find_best_split(X, y)

        if feature_idx is None or gain <= 0:
            node.is_leaf = True
            unique, counts = np.unique(y, return_counts=True)
            node.value = unique[np.argmax(counts)]
            return node

        left_X, left_y, right_X, right_y = self.split_dataset(X, y, feature_idx, threshold)

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self.build_tree(left_X, left_y, depth + 1)
        node.right = self.build_tree(right_X, right_y, depth + 1)

        return node

    def fit(self, X, y):
        """
        Train decision tree

        Args:
            X: training features, shape (n_samples, n_features)
            y: training labels, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.root = self.build_tree(X, y, depth=0)

        return self

    def predict_sample(self, x, node):
        """
        Predict single sample

        Args:
            x: single sample features
            node: current tree node

        Returns:
            prediction: predicted label
        """
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        """
        Predict labels for dataset

        Args:
            X: test features, shape (n_samples, n_features)

        Returns:
            predictions: predicted labels, shape (n_samples,)
        """
        X = np.array(X)
        predictions = np.array([self.predict_sample(x, self.root) for x in X])
        return predictions

    def score(self, X, y):
        """
        Calculate accuracy

        Args:
            X: test features
            y: test labels

        Returns:
            accuracy: accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class DecisionTreeRegressor:
    """
    Decision Tree Regressor for continuous values

    Splitting criterion: Mean Squared Error (MSE)
    MSE(D) = (1/|D|) * sum((yi - y_mean)^2)

    Used in gradient boosting algorithms (GBDT, XGBoost, etc.)
    """

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize decision tree regressor

        Args:
            max_depth: maximum depth of tree
            min_samples_split: minimum samples required to split node
            min_samples_leaf: minimum samples required at leaf node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def mse(self, y):
        """
        Calculate Mean Squared Error

        MSE = (1/n) * sum((yi - y_mean)^2)

        Args:
            y: target values

        Returns:
            mse: MSE value
        """
        if len(y) == 0:
            return 0

        return np.var(y)

    def split_dataset(self, X, y, feature_idx, threshold):
        """
        Split dataset by feature and threshold

        Args:
            X: features
            y: target values
            feature_idx: index of feature to split on
            threshold: threshold value

        Returns:
            left_X, left_y, right_X, right_y: split datasets
        """
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        return left_X, left_y, right_X, right_y

    def find_best_split(self, X, y):
        """
        Find best feature and threshold to split on

        Args:
            X: features, shape (n_samples, n_features)
            y: target values, shape (n_samples,)

        Returns:
            best_feature_idx: index of best feature
            best_threshold: best threshold value
            best_gain: best reduction in MSE
        """
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        current_mse = self.mse(y)

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_X, left_y, right_X, right_y = self.split_dataset(X, y, feature_idx, threshold)

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                left_mse = self.mse(left_y)
                right_mse = self.mse(right_y)

                n_left, n_right = len(left_y), len(right_y)
                weighted_mse = (n_left / n_samples) * left_mse + (n_right / n_samples) * right_mse

                gain = current_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        """
        Recursively build decision tree

        Args:
            X: features
            y: target values
            depth: current depth

        Returns:
            node: tree node
        """
        n_samples = len(y)

        node = DecisionTreeNode()

        if depth == self.max_depth or n_samples < self.min_samples_split:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        feature_idx, threshold, gain = self.find_best_split(X, y)

        if feature_idx is None or gain <= 0:
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        left_X, left_y, right_X, right_y = self.split_dataset(X, y, feature_idx, threshold)

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self.build_tree(left_X, left_y, depth + 1)
        node.right = self.build_tree(right_X, right_y, depth + 1)

        return node

    def fit(self, X, y):
        """
        Train decision tree

        Args:
            X: training features, shape (n_samples, n_features)
            y: training target values, shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        self.root = self.build_tree(X, y, depth=0)

        return self

    def predict_sample(self, x, node):
        """
        Predict single sample

        Args:
            x: single sample features
            node: current tree node

        Returns:
            prediction: predicted value
        """
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        """
        Predict values for dataset

        Args:
            X: test features, shape (n_samples, n_features)

        Returns:
            predictions: predicted values, shape (n_samples,)
        """
        X = np.array(X)
        predictions = np.array([self.predict_sample(x, self.root) for x in X])
        return predictions
