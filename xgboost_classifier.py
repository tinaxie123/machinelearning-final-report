import numpy as np
import pickle
from decision_tree import DecisionTreeNode
from metrics import classification_report, confusion_matrix, f1_score
import time


class XGBoostTree:
    def __init__(self, max_depth=6, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, min_child_weight=1):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root = None

    def calculate_leaf_weight(self, grad, hess):
        G = np.sum(grad)
        H = np.sum(hess)
        weight = -G / (H + self.reg_lambda)
        return weight

    def calculate_split_gain(self, grad_left, hess_left, grad_right, hess_right):
        GL = np.sum(grad_left)
        HL = np.sum(hess_left)
        GR = np.sum(grad_right)
        HR = np.sum(hess_right)

        gain = 0.5 * (
            (GL ** 2) / (HL + self.reg_lambda) +
            (GR ** 2) / (HR + self.reg_lambda) -
            ((GL + GR) ** 2) / (HL + HR + self.reg_lambda)
        ) - self.gamma

        return gain

    def find_best_split(self, X, grad, hess):
        n_samples, n_features = X.shape
        best_gain = -float('inf')
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(hess[left_mask]) < self.min_child_weight or np.sum(hess[right_mask]) < self.min_child_weight:
                    continue

                gain = self.calculate_split_gain(
                    grad[left_mask], hess[left_mask],
                    grad[right_mask], hess[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_gain

    def build_tree(self, X, grad, hess, depth=0):
        node = DecisionTreeNode()

        if depth >= self.max_depth or len(X) == 0:
            node.is_leaf = True
            node.value = self.calculate_leaf_weight(grad, hess)
            return node

        feature_idx, threshold, gain = self.find_best_split(X, grad, hess)

        if feature_idx is None or gain <= 0:
            node.is_leaf = True
            node.value = self.calculate_leaf_weight(grad, hess)
            return node

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self.build_tree(X[left_mask], grad[left_mask], hess[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], grad[right_mask], hess[right_mask], depth + 1)

        return node

    def fit(self, X, grad, hess):
        self.root = self.build_tree(X, grad, hess, depth=0)
        return self

    def predict_sample(self, x, node):
        if node.is_leaf:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


class XGBoostClassifier:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 reg_lambda=1.0, reg_alpha=0.0, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.trees_ = []
        self.classes_ = None
        self.n_classes_ = None
        self.init_pred_ = None

    def softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        print("Training XGBoost Classifier")

        X = np.array(X)
        y = np.array(y)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples = X.shape[0]

        print(f"Training set: {X.shape[0]} samples, {X.shape[1]} features, {self.n_classes_} classes")
        print(f"Parameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.learning_rate}")

        y_encoded = np.zeros((n_samples, self.n_classes_))
        for i, c in enumerate(self.classes_):
            y_encoded[y == c, i] = 1

        self.init_pred_ = np.log(np.mean(y_encoded, axis=0) + 1e-10)
        F = np.tile(self.init_pred_, (n_samples, 1))

        print("Training...")
        start_time = time.time()

        for m in range(self.n_estimators):
            proba = self.softmax(F)

            grad = proba - y_encoded
            hess = proba * (1 - proba)

            trees_m = []
            for k in range(self.n_classes_):
                tree = XGBoostTree(
                    max_depth=self.max_depth,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha
                )
                tree.fit(X, grad[:, k], hess[:, k])
                trees_m.append(tree)

            self.trees_.append(trees_m)

            for k in range(self.n_classes_):
                update = trees_m[k].predict(X)
                F[:, k] += self.learning_rate * update

            if (m + 1) % 20 == 0:
                print(f"  Progress: {m+1}/{self.n_estimators}")

        train_time = time.time() - start_time
        print(f"Training complete! Time: {train_time:.2f}s")

        return self

    def predict_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        F = np.tile(self.init_pred_, (n_samples, 1))

        for trees_m in self.trees_:
            for k in range(self.n_classes_):
                update = trees_m[k].predict(X)
                F[:, k] += self.learning_rate * update

        proba = self.softmax(F)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred_idx = np.argmax(proba, axis=1)
        return self.classes_[y_pred_idx]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def load_fused_features(file_path='features/fusion/fused_features.pkl'):
    print("Loading fused features")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = np.array(data['y_train'])
    y_val = np.array(data['y_val'])
    y_test = np.array(data['y_test'])

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Weights: {data['weights']}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    print("Model evaluation")

    results = {}
    for name, X, y in [('train', X_train, y_train),
                        ('val', X_val, y_val),
                        ('test', X_test, y_test)]:
        start_time = time.time()
        y_pred = model.predict(X)
        pred_time = time.time() - start_time

        acc = np.mean(y_pred == y)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')

        results[name] = {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

        print(f"[{name.upper()}] Acc: {acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}, Time: {pred_time:.2f}s")

    print("\nTest set classification report:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    return results


def main():
    print("XGBoost Classifier Experiment")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = XGBoostClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/xgboost_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/xgboost_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/xgboost_classifier.pkl")
    print(f"Results saved to: models/xgboost_results.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
