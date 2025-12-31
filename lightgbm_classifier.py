import numpy as np
import pickle
from decision_tree import DecisionTreeRegressor
from metrics import classification_report, confusion_matrix, f1_score
import time


class LightGBMClassifier:
    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1,
                 num_leaves=31, min_data_in_leaf=20, reg_lambda=0.0,
                 random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth > 0 else 100
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.reg_lambda = reg_lambda
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
        print("Training LightGBM Classifier")

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
            residuals = y_encoded - proba

            trees_m = []
            for k in range(self.n_classes_):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_data_in_leaf
                )
                tree.fit(X, residuals[:, k])
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
    print("LightGBM Classifier Experiment")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = LightGBMClassifier(
        n_estimators=100,
        max_depth=-1,
        learning_rate=0.1,
        num_leaves=31,
        min_data_in_leaf=20,
        reg_lambda=0.0,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/lightgbm_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/lightgbm_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/lightgbm_classifier.pkl")
    print(f"Results saved to: models/lightgbm_results.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
