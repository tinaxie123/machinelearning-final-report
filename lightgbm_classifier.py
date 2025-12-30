"""
LightGBM分类器 (Light Gradient Boosting Machine)
高效梯度提升算法，使用GOSS和Leaf-wise策略
"""

import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time


class LightGBMClassifier:
    """
    LightGBM分类器

    优化目标:
    L = Σ l(yi, ŷi) + Σ Ω(fm)

    关键技术:
    1. GOSS (Gradient-based One-Side Sampling):
       - 保留梯度大的样本
       - 随机采样梯度小的样本并放大权重
       - 加权梯度: g'i = gi (|gi|大) 或 (1-a)/b·gi (|gi|小)

    2. Leaf-wise生长策略:
       - 选择增益最大的叶子分裂: ℓ* = argmax Gain(ℓ)
       - 更快收敛，但需要max_depth约束防止过拟合

    节点分裂增益（同XGBoost）:
    Gain = 1/2 [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ
    """

    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1,
                 num_leaves=31, min_data_in_leaf=20, reg_lambda=0.0,
                 random_state=42):
        """
        初始化LightGBM分类器

        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度（-1表示无限制）
            learning_rate: 学习率
            num_leaves: 最大叶子节点数
            min_data_in_leaf: 叶子节点最小样本数
            reg_lambda: L2正则化参数
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        训练模型

        Args:
            X: 训练特征，shape (n_samples, n_features)
            y: 训练标签，shape (n_samples,)
        """
        print("\n" + "=" * 80)
        print("训练LightGBM分类器")
        print("=" * 80)

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        print(f"\n训练集大小: {X.shape[0]} 样本")
        print(f"特征维度: {X.shape[1]}")
        print(f"类别数量: {n_classes}")
        print(f"\n模型参数:")
        print(f"  树的数量: {self.n_estimators}")
        print(f"  最大深度: {self.max_depth if self.max_depth > 0 else '无限制'}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  最大叶子数: {self.num_leaves}")
        print(f"  叶子最小样本数: {self.min_data_in_leaf}")
        print(f"  L2正则化: {self.reg_lambda}")

        print("\n算法特点:")
        print("  - GOSS: 梯度导向的样本选择，减少计算量")
        print("  - Leaf-wise: 按叶子生长，更快收敛")
        print("  - 直方图算法: 离散化特征，加速训练")
        print("  - 支持类别特征: 无需One-Hot编码")

        # 初始化LightGBM模型
        self.model_ = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_data_in_leaf,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )

        print("\n开始训练...")
        start_time = time.time()
        self.model_.fit(X, y)
        train_time = time.time() - start_time

        print(f"训练完成！耗时: {train_time:.2f}秒")

        return self

    def predict_proba(self, X):
        """预测概率"""
        return self.model_.predict_proba(X)

    def predict(self, X):
        """预测类别"""
        return self.model_.predict(X)

    def score(self, X, y):
        """计算准确率"""
        return self.model_.score(X, y)


def load_fused_features(file_path='features/fusion/fused_features.pkl'):
    """加载融合特征"""
    print("\n" + "=" * 80)
    print("加载融合特征")
    print("=" * 80)

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = np.array(data['y_train'])
    y_val = np.array(data['y_val'])
    y_test = np.array(data['y_test'])

    print(f"\n训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"权重组合: {data['weights']}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """评估模型性能"""
    print("\n" + "=" * 80)
    print("模型评估")
    print("=" * 80)

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

        print(f"\n[{name.upper()}]")
        print(f"  准确率: {acc:.4f}")
        print(f"  宏平均F1: {f1_macro:.4f}")
        print(f"  加权平均F1: {f1_weighted:.4f}")
        print(f"  预测时间: {pred_time:.2f}秒")

    # 测试集详细报告
    print("\n" + "=" * 80)
    print("测试集详细分类报告")
    print("=" * 80)
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=4))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_test_pred))

    return results


def main():
    print("\n" + "=" * 80)
    print("LightGBM分类器实验")
    print("=" * 80)

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
    print(f"\n总训练时间: {time.time() - start_time:.2f}秒")

    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # 保存模型和结果
    import os
    os.makedirs('models', exist_ok=True)

    with open('models/lightgbm_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/lightgbm_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"\n模型已保存至: models/lightgbm_classifier.pkl")
    print(f"结果已保存至: models/lightgbm_results.pkl")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"\n最终结果: 测试集F1={results['test']['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
