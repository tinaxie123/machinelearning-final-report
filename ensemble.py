"""
集成学习模块：加权软投票与 Stacking 集成
"""

import numpy as np
import pickle
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from base_models import BaseModelManager
from config import ENSEMBLE_CONFIG


class WeightedSoftVoting:
    """加权软投票集成"""

    def __init__(self, base_model_manager: BaseModelManager):
        """
        初始化加权软投票

        Args:
            base_model_manager: 基模型管理器
        """
        self.base_model_manager = base_model_manager
        self.weights = {}

    def calculate_weights(self, X_val: np.ndarray, y_val: List[str]):
        """
        根据验证集 F1 值计算权重

        Args:
            X_val: 验证集特征
            y_val: 验证集标签
        """
        print("\n" + "=" * 60)
        print("计算基模型权重...")
        print("=" * 60)

        f1_scores = {}

        # 计算每个模型的宏平均 F1 值
        for model_name in self.base_model_manager.get_model_names():
            y_pred = self.base_model_manager.predict_single_model(model_name, X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            f1_scores[model_name] = f1
            print(f"{model_name}: F1 = {f1:.4f}")

        # 归一化权重
        total_f1 = sum(f1_scores.values())
        self.weights = {name: f1 / total_f1 for name, f1 in f1_scores.items()}

        print("\n权重分配:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.4f}")
        print("=" * 60)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        加权软投票预测

        Args:
            X: 特征矩阵

        Returns:
            预测标签
        """
        if not self.weights:
            raise ValueError("请先调用 calculate_weights() 方法计算权重")

        # 获取所有模型的概率预测
        all_probas = self.base_model_manager.predict_proba_all_models(X)

        # 加权平均
        weighted_proba = np.zeros_like(list(all_probas.values())[0])

        for model_name, proba in all_probas.items():
            weighted_proba += self.weights[model_name] * proba

        # 选择概率最大的类别
        y_pred_encoded = np.argmax(weighted_proba, axis=1)
        y_pred = self.base_model_manager.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        加权软投票预测概率

        Args:
            X: 特征矩阵

        Returns:
            类别概率矩阵
        """
        if not self.weights:
            raise ValueError("请先调用 calculate_weights() 方法计算权重")

        # 获取所有模型的概率预测
        all_probas = self.base_model_manager.predict_proba_all_models(X)

        # 加权平均
        weighted_proba = np.zeros_like(list(all_probas.values())[0])

        for model_name, proba in all_probas.items():
            weighted_proba += self.weights[model_name] * proba

        return weighted_proba


class StackingEnsemble:
    """Stacking 集成（使用 OOF 交叉验证）"""

    def __init__(self, base_model_manager: BaseModelManager, n_folds=5,
                 meta_model_config=None):
        """
        初始化 Stacking 集成

        Args:
            base_model_manager: 基模型管理器
            n_folds: 交叉验证折数
            meta_model_config: 元学习器配置
        """
        self.base_model_manager = base_model_manager
        self.n_folds = n_folds

        # 元学习器
        if meta_model_config is None:
            meta_model_config = ENSEMBLE_CONFIG['meta_model']

        self.meta_model = LogisticRegression(**meta_model_config)

    def _generate_oof_features(self, X_train: np.ndarray, y_train: List[str]) -> np.ndarray:
        """
        生成 Out-of-Fold (OOF) 元特征

        Args:
            X_train: 训练集特征
            y_train: 训练集标签

        Returns:
            OOF 元特征矩阵
        """
        print("\n" + "=" * 60)
        print("生成 OOF 元特征...")
        print("=" * 60)

        n_samples = X_train.shape[0]
        n_classes = len(self.base_model_manager.label_encoder.classes_)
        n_models = len(self.base_model_manager.get_model_names())

        # 初始化 OOF 元特征矩阵
        oof_meta_features = np.zeros((n_samples, n_models * n_classes))

        # K 折交叉验证
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\n处理第 {fold + 1}/{self.n_folds} 折...")

            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = [y_train[i] for i in train_idx]

            # 训练每个基模型
            for i, model_name in enumerate(self.base_model_manager.get_model_names()):
                self.base_model_manager.train_single_model(
                    model_name, X_fold_train, y_fold_train, verbose=False
                )

                # 预测验证集概率
                proba = self.base_model_manager.predict_proba_single_model(model_name, X_fold_val)

                # 存储到 OOF 矩阵
                oof_meta_features[val_idx, i * n_classes:(i + 1) * n_classes] = proba

            print(f"第 {fold + 1} 折完成")

        print("\nOOF 元特征生成完成！")
        print(f"元特征形状: {oof_meta_features.shape}")
        print("=" * 60)

        return oof_meta_features

    def _generate_test_meta_features(self, X_test: np.ndarray) -> np.ndarray:
        """
        生成测试集元特征

        Args:
            X_test: 测试集特征

        Returns:
            测试集元特征矩阵
        """
        n_classes = len(self.base_model_manager.label_encoder.classes_)
        n_models = len(self.base_model_manager.get_model_names())

        test_meta_features = np.zeros((X_test.shape[0], n_models * n_classes))

        # 获取所有基模型的概率预测
        all_probas = self.base_model_manager.predict_proba_all_models(X_test)

        for i, model_name in enumerate(self.base_model_manager.get_model_names()):
            proba = all_probas[model_name]
            test_meta_features[:, i * n_classes:(i + 1) * n_classes] = proba

        return test_meta_features

    def fit(self, X_train: np.ndarray, y_train: List[str]):
        """
        训练 Stacking 集成模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
        """
        print("\n" + "=" * 60)
        print("开始训练 Stacking 集成模型...")
        print("=" * 60)

        # 生成 OOF 元特征
        oof_meta_features = self._generate_oof_features(X_train, y_train)

        # 训练元学习器
        print("\n训练元学习器...")
        y_train_encoded = self.base_model_manager.label_encoder.transform(y_train)
        self.meta_model.fit(oof_meta_features, y_train_encoded)

        print("Stacking 集成模型训练完成！")
        print("=" * 60)

        # 重新在全部训练集上训练所有基模型
        print("\n重新在全部训练集上训练基模型...")
        self.base_model_manager.train_all_models(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Stacking 预测

        Args:
            X: 特征矩阵

        Returns:
            预测标签
        """
        # 生成测试集元特征
        test_meta_features = self._generate_test_meta_features(X)

        # 元学习器预测
        y_pred_encoded = self.meta_model.predict(test_meta_features)
        y_pred = self.base_model_manager.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Stacking 预测概率

        Args:
            X: 特征矩阵

        Returns:
            类别概率矩阵
        """
        # 生成测试集元特征
        test_meta_features = self._generate_test_meta_features(X)

        # 元学习器预测概率
        y_proba = self.meta_model.predict_proba(test_meta_features)

        return y_proba


class EnsembleManager:
    """集成学习管理器：统一管理加权软投票与 Stacking"""

    def __init__(self, base_model_manager: BaseModelManager, n_folds=5):
        """
        初始化集成学习管理器

        Args:
            base_model_manager: 基模型管理器
            n_folds: Stacking 交叉验证折数
        """
        self.base_model_manager = base_model_manager
        self.weighted_voting = WeightedSoftVoting(base_model_manager)
        self.stacking = StackingEnsemble(base_model_manager, n_folds)

    def train(self, X_train: np.ndarray, y_train: List[str],
              X_val: np.ndarray, y_val: List[str]):
        """
        训练集成模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
        """
        # 训练基模型
        self.base_model_manager.train_all_models(X_train, y_train)

        # 计算加权软投票权重
        self.weighted_voting.calculate_weights(X_val, y_val)

        # 训练 Stacking
        self.stacking.fit(X_train, y_train)

    def predict_weighted_voting(self, X: np.ndarray) -> np.ndarray:
        """加权软投票预测"""
        return self.weighted_voting.predict(X)

    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Stacking 预测"""
        return self.stacking.predict(X)

    def save(self, path: str):
        """保存集成模型"""
        ensemble_data = {
            'weighted_voting': self.weighted_voting,
            'stacking': self.stacking
        }
        with open(path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        print(f"集成模型已保存至: {path}")

    def load(self, path: str):
        """加载集成模型"""
        with open(path, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.weighted_voting = ensemble_data['weighted_voting']
        self.stacking = ensemble_data['stacking']
        print(f"集成模型已从 {path} 加载")


if __name__ == '__main__':
    # 测试集成学习
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 生成模拟数据
    X, y = make_classification(n_samples=500, n_features=50, n_informative=30,
                                n_classes=3, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 转换标签
    label_map = {0: '娱乐', 1: '财经', 2: '教育'}
    y_train = [label_map[label] for label in y_train]
    y_val = [label_map[label] for label in y_val]
    y_test = [label_map[label] for label in y_test]

    # 初始化基模型管理器
    base_manager = BaseModelManager()

    # 初始化集成管理器
    ensemble_manager = EnsembleManager(base_manager, n_folds=3)

    # 训练
    ensemble_manager.train(X_train, y_train, X_val, y_val)

    # 预测
    print("\n" + "=" * 60)
    print("测试预测...")
    print("=" * 60)

    y_pred_voting = ensemble_manager.predict_weighted_voting(X_test[:5])
    print(f"加权软投票预测: {y_pred_voting}")

    y_pred_stacking = ensemble_manager.predict_stacking(X_test[:5])
    print(f"Stacking 预测: {y_pred_stacking}")
