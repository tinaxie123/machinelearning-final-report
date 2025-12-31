import numpy as np
import pickle
from metrics import classification_report, confusion_matrix, f1_score
import time


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y):
        print("Training K-Nearest Neighbors Classifier")

        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(y)

        print(f"Training set: {self.X_train_.shape[0]} samples, {self.X_train_.shape[1]} features, {len(self.classes_)} classes")
        print(f"Number of neighbors K: {self.n_neighbors}")
        print(f"Distance metric: Cosine similarity")
        print("KNN training complete (stored training data)")

        return self

    def cosine_similarity(self, X_test):
        X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10)
        X_train_norm = self.X_train_ / (np.linalg.norm(self.X_train_, axis=1, keepdims=True) + 1e-10)
        similarities = X_test_norm @ X_train_norm.T
        return similarities

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]

        similarities = self.cosine_similarity(X)

        neighbor_indices = np.argsort(similarities, axis=1)[:, -self.n_neighbors:]

        neighbor_labels = self.y_train_[neighbor_indices]

        y_pred = []
        for labels in neighbor_labels:
            unique, counts = np.unique(labels, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])

        y_pred = np.array(y_pred)

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
    print("K-Nearest Neighbors Classifier Experiment")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = KNNClassifier(n_neighbors=5)

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/knn_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/knn_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/knn_classifier.pkl")
    print(f"Results saved to: models/knn_results.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
