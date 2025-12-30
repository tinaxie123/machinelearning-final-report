"""
多项式朴素贝叶斯分类器 (Multinomial Naive Bayes)
从底层实现，用于文本分类任务
"""

import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time


class MultinomialNaiveBayes:
    """
    多项式朴素贝叶斯分类器

    基于贝叶斯定理: P(C|x) ∝ P(C) * P(x|C)
    假设特征条件独立: P(x|C) ≈ ∏P(xi|C)
    """

    def __init__(self, alpha=1.0):
        """
        初始化多项式朴素贝叶斯分类器

        Args:
            alpha: 拉普拉斯平滑系数，默认1.0
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None  # P(Ck)
        self.feature_log_prob_ = None  # log P(xi|Ck)
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """
        训练模型

        Args:
            X: 训练特征，shape (n_samples, n_features)
            y: 训练标签，shape (n_samples,)
        """
        print("\n" + "=" * 80)
        print("训练多项式朴素贝叶斯分类器")
        print("=" * 80)

        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)

        self.n_features_ = X.shape[1]

        # 获取所有类别
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        print(f"\n训练集大小: {X.shape[0]} 样本")
        print(f"特征维度: {self.n_features_}")
        print(f"类别数量: {self.n_classes_}")
        print(f"拉普拉斯平滑系数 α: {self.alpha}")

        # 计算先验概率 P(Ck)
        self.class_prior_ = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            self.class_prior_[i] = np.sum(y == c) / len(y)

        print(f"\n先验概率 P(Ck):")
        for i, c in enumerate(self.classes_):
            print(f"  类别 {c}: {self.class_prior_[i]:.4f}")

        # 计算条件概率 P(xi|Ck)
        # 由于特征可能是负数（标准化后的特征），需要先进行处理
        # 将特征平移到非负区间
        X_shifted = X - X.min() + 1e-10

        self.feature_log_prob_ = np.zeros((self.n_classes_, self.n_features_))

        print("\n计算条件概率 P(xi|Ck)...")
        for i, c in enumerate(self.classes_):
            # 获取类别c的所有样本
            X_c = X_shifted[y == c]

            # 计算类别c下每个特征的总计数
            # Nik: 类别Ck下词项i的总计数
            feature_count = X_c.sum(axis=0)

            # 应用拉普拉斯平滑
            # P(xi|Ck) = (Nik + α) / (Σj Njk + α * n)
            numerator = feature_count + self.alpha
            denominator = feature_count.sum() + self.alpha * self.n_features_

            # 存储log概率（避免下溢）
            self.feature_log_prob_[i, :] = np.log(numerator / denominator)

        print("训练完成！")

        return self

    def predict_log_proba(self, X):
        """
        预测对数概率

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            log_proba: 对数概率，shape (n_samples, n_classes)
        """
        X = np.array(X)

        # 平移特征到非负区间（与训练时保持一致）
        X_shifted = X - X.min() + 1e-10

        # 计算后验概率的对数
        # log P(C|x) = log P(C) + Σ xi * log P(xi|C)
        log_proba = np.zeros((X_shifted.shape[0], self.n_classes_))

        for i in range(self.n_classes_):
            # log P(Ck)
            log_prior = np.log(self.class_prior_[i])

            # Σ xi * log P(xi|Ck)
            log_likelihood = X_shifted @ self.feature_log_prob_[i, :]

            log_proba[:, i] = log_prior + log_likelihood

        return log_proba

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            proba: 概率，shape (n_samples, n_classes)
        """
        log_proba = self.predict_log_proba(X)

        # 归一化（使用log-sum-exp技巧避免数值下溢）
        log_proba_max = log_proba.max(axis=1, keepdims=True)
        exp_log_proba = np.exp(log_proba - log_proba_max)
        proba = exp_log_proba / exp_log_proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """
        预测类别

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            y_pred: 预测标签，shape (n_samples,)
        """
        log_proba = self.predict_log_proba(X)

        # 选择概率最大的类别
        # C = argmax_Ck [log P(Ck) + Σ xi * log P(xi|Ck)]
        y_pred_idx = np.argmax(log_proba, axis=1)
        y_pred = self.classes_[y_pred_idx]

        return y_pred

    def score(self, X, y):
        """
        计算准确率

        Args:
            X: 测试特征
            y: 测试标签

        Returns:
            accuracy: 准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


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

    # 训练集评估
    print("\n[1/3] 训练集评估...")
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    train_time = time.time() - start_time

    train_acc = np.mean(y_train_pred == y_train)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')

    print(f"  准确率: {train_acc:.4f}")
    print(f"  宏平均F1: {train_f1_macro:.4f}")
    print(f"  加权平均F1: {train_f1_weighted:.4f}")
    print(f"  预测时间: {train_time:.2f}秒")

    # 验证集评估
    print("\n[2/3] 验证集评估...")
    start_time = time.time()
    y_val_pred = model.predict(X_val)
    val_time = time.time() - start_time

    val_acc = np.mean(y_val_pred == y_val)
    val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
    val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')

    print(f"  准确率: {val_acc:.4f}")
    print(f"  宏平均F1: {val_f1_macro:.4f}")
    print(f"  加权平均F1: {val_f1_weighted:.4f}")
    print(f"  预测时间: {val_time:.2f}秒")

    # 测试集评估
    print("\n[3/3] 测试集评估...")
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_time = time.time() - start_time

    test_acc = np.mean(y_test_pred == y_test)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')

    print(f"  准确率: {test_acc:.4f}")
    print(f"  宏平均F1: {test_f1_macro:.4f}")
    print(f"  加权平均F1: {test_f1_weighted:.4f}")
    print(f"  预测时间: {test_time:.2f}秒")

    # 详细分类报告（测试集）
    print("\n" + "=" * 80)
    print("测试集详细分类报告")
    print("=" * 80)
    print(classification_report(y_test, y_test_pred, digits=4))

    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    return {
        'train': {'accuracy': train_acc, 'f1_macro': train_f1_macro, 'f1_weighted': train_f1_weighted},
        'val': {'accuracy': val_acc, 'f1_macro': val_f1_macro, 'f1_weighted': val_f1_weighted},
        'test': {'accuracy': test_acc, 'f1_macro': test_f1_macro, 'f1_weighted': test_f1_weighted}
    }


def main():
    print("\n" + "=" * 80)
    print("多项式朴素贝叶斯分类器实验")
    print("=" * 80)

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    # 2. 训练模型
    start_time = time.time()
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\n总训练时间: {train_time:.2f}秒")

    # 3. 评估模型
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. 保存模型
    model_path = 'models/mnb_classifier.pkl'
    import os
    os.makedirs('models', exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n模型已保存至: {model_path}")

    # 5. 保存结果
    results_path = 'models/mnb_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"结果已保存至: {results_path}")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print("\n最终结果摘要:")
    print(f"  测试集准确率: {results['test']['accuracy']:.4f}")
    print(f"  测试集宏平均F1: {results['test']['f1_macro']:.4f}")
    print(f"  测试集加权平均F1: {results['test']['f1_weighted']:.4f}")


if __name__ == '__main__':
    main()
