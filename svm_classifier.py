import numpy as np
import pickle
from metrics import classification_report, confusion_matrix, f1_score
import time
from itertools import combinations


class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.001, max_iter=1000, random_state=42):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * (1 / self.max_iter) * self.w)
                else:
                    self.w -= self.learning_rate * (2 * (1 / self.max_iter) * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

        return self

    def predict(self, X):
        X = np.array(X)
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)


class SVMClassifierOVO:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', random_state=42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.classes_ = None
        self.classifiers_ = {}
        self.class_pairs_ = []

    def fit(self, X, y):
        print("Training SVM Classifier (OVO strategy)")

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        print(f"Training set: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
        print(f"Kernel: {self.kernel}, C: {self.C}")

        self.class_pairs_ = list(combinations(self.classes_, 2))
        n_classifiers = len(self.class_pairs_)

        print(f"OVO strategy: training {n_classifiers} binary classifiers")
        print(f"  Formula: K(K-1)/2 = {n_classes}x{n_classes-1}/2 = {n_classifiers}")

        print("Training binary classifiers...")
        start_time = time.time()

        for idx, (class_i, class_j) in enumerate(self.class_pairs_, 1):
            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]

            y_binary = np.where(y_pair == class_i, 1, -1)

            clf = LinearSVM(
                C=self.C,
                learning_rate=0.001,
                max_iter=1000,
                random_state=self.random_state
            )
            clf.fit(X_pair, y_binary)

            self.classifiers_[(class_i, class_j)] = clf

            if idx % 2 == 0 or idx == n_classifiers:
                print(f"  Progress: {idx}/{n_classifiers}")

        train_time = time.time() - start_time
        print(f"Training complete! Time: {train_time:.2f}s")

        return self

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        votes = np.zeros((n_samples, n_classes))

        for (class_i, class_j), clf in self.classifiers_.items():
            predictions = clf.predict(X)

            idx_i = np.where(self.classes_ == class_i)[0][0]
            idx_j = np.where(self.classes_ == class_j)[0][0]

            votes[predictions == 1, idx_i] += 1
            votes[predictions == -1, idx_j] += 1

        y_pred_idx = np.argmax(votes, axis=1)
        y_pred = self.classes_[y_pred_idx]

        return y_pred

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
    print("SVM Classifier Experiment (OVO strategy)")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = SVMClassifierOVO(
        kernel='linear',
        C=1.0,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/svm_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/svm_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/svm_classifier.pkl")
    print(f"Results saved to: models/svm_results.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
