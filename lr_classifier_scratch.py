import numpy as np
import pickle
from metrics import classification_report, confusion_matrix, f1_score
import time


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, max_iter=1000, reg_lambda=0.01,
                 tol=1e-4, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.random_state = random_state
        self.W_ = None
        self.b_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.loss_history_ = []

    def softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, X, y_onehot):
        n_samples = X.shape[0]

        z = X @ self.W_ + self.b_
        proba = self.softmax(z)

        cross_entropy = -np.sum(y_onehot * np.log(proba + 1e-10)) / n_samples

        reg_loss = 0.5 * self.reg_lambda * np.sum(self.W_ ** 2)

        total_loss = cross_entropy + reg_loss

        return total_loss

    def fit(self, X, y):
        print("Training Logistic Regression Classifier (Softmax)")

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        print(f"Training set: {n_samples} samples, {n_features} features, {self.n_classes_} classes")
        print(f"Parameters: lr={self.learning_rate}, L2_reg={self.reg_lambda}, max_iter={self.max_iter}, tol={self.tol}")
        print("Algorithm: Batch Gradient Descent")
        print("Objective: Cross-entropy loss + L2 regularization")

        y_onehot = np.zeros((n_samples, self.n_classes_))
        for i, label in enumerate(y):
            class_idx = np.where(self.classes_ == label)[0][0]
            y_onehot[i, class_idx] = 1

        np.random.seed(self.random_state)
        self.W_ = np.random.randn(n_features, self.n_classes_) * 0.01
        self.b_ = np.zeros(self.n_classes_)

        print("Training...")
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            z = X @ self.W_ + self.b_
            proba = self.softmax(z)

            loss = self.compute_loss(X, y_onehot)
            self.loss_history_.append(loss)

            error = proba - y_onehot

            grad_W = (X.T @ error) / n_samples + self.reg_lambda * self.W_
            grad_b = np.sum(error, axis=0) / n_samples

            self.W_ -= self.learning_rate * grad_W
            self.b_ -= self.learning_rate * grad_b

            if (iteration + 1) % 100 == 0 or iteration == 0:
                print(f"  Iteration {iteration+1}/{self.max_iter}, Loss: {loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                print(f"Converged at iteration {iteration+1}")
                break

            prev_loss = loss

        print(f"Training complete! Final loss: {self.loss_history_[-1]:.6f}")

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        z = X @ self.W_ + self.b_
        proba = self.softmax(z)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        y_pred_idx = np.argmax(proba, axis=1)
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
    print("Logistic Regression Classifier Experiment (Softmax)")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = LogisticRegressionClassifier(
        learning_rate=0.1,
        max_iter=1000,
        reg_lambda=0.01,
        tol=1e-4,
        random_state=42
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/lr_classifier_scratch.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/lr_results_scratch.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/lr_classifier_scratch.pkl")
    print(f"Results saved to: models/lr_results_scratch.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
