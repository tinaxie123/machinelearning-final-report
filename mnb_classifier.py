import numpy as np
import pickle
from metrics import classification_report, confusion_matrix, f1_score
import time


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        print("Training Multinomial Naive Bayes Classifier")

        X = np.array(X)
        y = np.array(y)

        self.n_features_ = X.shape[1]

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        print(f"Training set: {X.shape[0]} samples, {self.n_features_} features, {self.n_classes_} classes")
        print(f"Laplace smoothing alpha: {self.alpha}")

        self.class_prior_ = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            self.class_prior_[i] = np.sum(y == c) / len(y)

        print("Class priors P(Ck):")
        for i, c in enumerate(self.classes_):
            print(f"  Class {c}: {self.class_prior_[i]:.4f}")

        X_shifted = X - X.min() + 1e-10

        self.feature_log_prob_ = np.zeros((self.n_classes_, self.n_features_))

        print("Computing conditional probabilities P(xi|Ck)...")
        for i, c in enumerate(self.classes_):
            X_c = X_shifted[y == c]

            feature_count = X_c.sum(axis=0)

            numerator = feature_count + self.alpha
            denominator = feature_count.sum() + self.alpha * self.n_features_

            self.feature_log_prob_[i, :] = np.log(numerator / denominator)

        print("Training complete!")

        return self

    def predict_log_proba(self, X):
        X = np.array(X)

        X_shifted = X - X.min() + 1e-10

        log_proba = np.zeros((X_shifted.shape[0], self.n_classes_))

        for i in range(self.n_classes_):
            log_prior = np.log(self.class_prior_[i])

            log_likelihood = X_shifted @ self.feature_log_prob_[i, :]

            log_proba[:, i] = log_prior + log_likelihood

        return log_proba

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)

        log_proba_max = log_proba.max(axis=1, keepdims=True)
        exp_log_proba = np.exp(log_proba - log_proba_max)
        proba = exp_log_proba / exp_log_proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)

        y_pred_idx = np.argmax(log_proba, axis=1)
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
    print("Multinomial Naive Bayes Classifier Experiment")

    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    model = MultinomialNaiveBayes(alpha=1.0)

    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Total training time: {time.time() - start_time:.2f}s")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    import os
    os.makedirs('models', exist_ok=True)

    with open('models/mnb_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/mnb_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\nModel saved to: models/mnb_classifier.pkl")
    print(f"Results saved to: models/mnb_results.pkl")
    print(f"Final result: Test F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
