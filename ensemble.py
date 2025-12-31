import numpy as np
import pickle
from typing import Dict, List
from metrics import f1_score
from lr_classifier_scratch import LogisticRegressionClassifier


class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop


class WeightedSoftVoting:
    def __init__(self, base_model_manager):
        self.base_model_manager = base_model_manager
        self.weights = {}

    def calculate_weights(self, X_val: np.ndarray, y_val: List[str]):
        print("Calculating base model weights...")

        f1_scores = {}

        for model_name in self.base_model_manager.get_model_names():
            y_pred = self.base_model_manager.predict_single_model(model_name, X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            f1_scores[model_name] = f1
            print(f"{model_name}: F1 = {f1:.4f}")

        total_f1 = sum(f1_scores.values())
        self.weights = {name: f1 / total_f1 for name, f1 in f1_scores.items()}

        print("\nWeight allocation:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.weights:
            raise ValueError("Please call calculate_weights() first")

        all_probas = self.base_model_manager.predict_proba_all_models(X)

        weighted_proba = np.zeros_like(list(all_probas.values())[0])

        for model_name, proba in all_probas.items():
            weighted_proba += self.weights[model_name] * proba

        y_pred_encoded = np.argmax(weighted_proba, axis=1)
        y_pred = self.base_model_manager.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.weights:
            raise ValueError("Please call calculate_weights() first")

        all_probas = self.base_model_manager.predict_proba_all_models(X)

        weighted_proba = np.zeros_like(list(all_probas.values())[0])

        for model_name, proba in all_probas.items():
            weighted_proba += self.weights[model_name] * proba

        return weighted_proba


class StackingEnsemble:
    def __init__(self, base_model_manager, n_folds=5):
        self.base_model_manager = base_model_manager
        self.n_folds = n_folds

        self.meta_model = LogisticRegressionClassifier(
            learning_rate=0.1,
            max_iter=1000,
            reg_lambda=0.01,
            random_state=42
        )

    def _generate_oof_features(self, X_train: np.ndarray, y_train: List[str]) -> np.ndarray:
        print("Generating OOF meta-features...")

        n_samples = X_train.shape[0]
        n_classes = len(self.base_model_manager.label_encoder.classes_)
        n_models = len(self.base_model_manager.get_model_names())

        oof_meta_features = np.zeros((n_samples, n_models * n_classes))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\nProcessing fold {fold + 1}/{self.n_folds}...")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = [y_train[i] for i in train_idx]

            for i, model_name in enumerate(self.base_model_manager.get_model_names()):
                self.base_model_manager.train_single_model(
                    model_name, X_fold_train, y_fold_train, verbose=False
                )

                proba = self.base_model_manager.predict_proba_single_model(model_name, X_fold_val)

                oof_meta_features[val_idx, i * n_classes:(i + 1) * n_classes] = proba

            print(f"Fold {fold + 1} complete")

        print(f"\nOOF meta-features generated! Shape: {oof_meta_features.shape}")

        return oof_meta_features

    def _generate_test_meta_features(self, X_test: np.ndarray) -> np.ndarray:
        n_classes = len(self.base_model_manager.label_encoder.classes_)
        n_models = len(self.base_model_manager.get_model_names())

        test_meta_features = np.zeros((X_test.shape[0], n_models * n_classes))

        all_probas = self.base_model_manager.predict_proba_all_models(X_test)

        for i, model_name in enumerate(self.base_model_manager.get_model_names()):
            proba = all_probas[model_name]
            test_meta_features[:, i * n_classes:(i + 1) * n_classes] = proba

        return test_meta_features

    def fit(self, X_train: np.ndarray, y_train: List[str]):
        print("Training Stacking ensemble model...")

        oof_meta_features = self._generate_oof_features(X_train, y_train)

        print("\nTraining meta-learner...")
        y_train_array = np.array(y_train)
        self.meta_model.fit(oof_meta_features, y_train_array)

        print("Stacking ensemble model training complete!")

        print("\nRetraining base models on full training set...")
        self.base_model_manager.train_all_models(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        test_meta_features = self._generate_test_meta_features(X)

        y_pred = self.meta_model.predict(test_meta_features)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        test_meta_features = self._generate_test_meta_features(X)

        y_proba = self.meta_model.predict_proba(test_meta_features)

        return y_proba


class EnsembleManager:
    def __init__(self, base_model_manager, n_folds=5):
        self.base_model_manager = base_model_manager
        self.weighted_voting = WeightedSoftVoting(base_model_manager)
        self.stacking = StackingEnsemble(base_model_manager, n_folds)

    def train(self, X_train: np.ndarray, y_train: List[str],
              X_val: np.ndarray, y_val: List[str]):
        self.base_model_manager.train_all_models(X_train, y_train)

        self.weighted_voting.calculate_weights(X_val, y_val)

        self.stacking.fit(X_train, y_train)

    def predict_weighted_voting(self, X: np.ndarray) -> np.ndarray:
        return self.weighted_voting.predict(X)

    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        return self.stacking.predict(X)

    def save(self, path: str):
        ensemble_data = {
            'weighted_voting': self.weighted_voting,
            'stacking': self.stacking
        }
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        print(f"Ensemble model saved to: {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.weighted_voting = ensemble_data['weighted_voting']
        self.stacking = ensemble_data['stacking']
        print(f"Ensemble model loaded from {path}")
