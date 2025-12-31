import numpy as np
import pickle
from decision_tree import DecisionTreeClassifier
from metrics import classification_report, confusion_matrix, f1_score
import time


class DecisionTreeWithFeatureSubset(DecisionTreeClassifier):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=42):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape

        if self.max_features is not None and self.max_features < n_features:
            np.random.seed(self.random_state)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)

        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        current_gini = self.gini_impurity(y)

        for feature_idx in feature_indices:
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


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.classes_ = None

    def fit(self, X, y):
        print("Training Random Forest Classifier")

        X = np.array(X)
        y = np.array(y)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features

        print(f"Training set: {n_samples} samples, {n_features} features, {n_classes} classes")
        print(f"Parameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}, max_features={max_features}")

        print("Training...")
        start_time = time.time()

        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTreeWithFeatureSubset(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + i
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees_.append(tree)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{self.n_estimators}")

        train_time = time.time() - start_time
        print(f"Training complete! Time: {train_time:.2f}s")

        return self

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        predictions = np.zeros((n_samples, len(self.trees_)))
        for i, tree in enumerate(self.trees_):
            predictions[:, i] = tree.predict(X)

        y_pred = []
        for i in range(n_samples):
            unique, counts = np.unique(predictions[i], return_counts=True)
            y_pred.append(unique[np.argmax(counts)])

        return np.array(y_pred)

    def predict_proba(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        proba = np.zeros((n_samples, n_classes))

        for tree in self.trees_:
            pred = tree.predict(X)
            for i, c in enumerate(self.classes_):
                proba[:, i] += (pred == c)

        proba /= len(self.trees_)
        return proba

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
    print("Random Forest Classifier Experiment")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features='sqrt',
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/rf_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/rf_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/rf_classifier.pkl")
    print(f"Results saved to: models/rf_results.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
