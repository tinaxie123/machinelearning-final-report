"""
支持向量机分类器 (Support Vector Machine with OVO)
使用One-vs-One策略实现多分类
"""

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time
from itertools import combinations


class SVMClassifierOVO:
    """
    支持向量机分类器（One-vs-One多分类策略）

    对K个类别，训练K(K-1)/2个二分类器
    每个分类器fij(x)区分类别ci和cj
    预测时通过投票选择最终类别
    """

    def __init__(self, kernel='linear', C=1.0, gamma='scale', random_state=42):
        """
        初始化SVM分类器

        Args:
            kernel: 核函数类型 ('linear', 'rbf', 'poly', 'sigmoid')
            C: 正则化参数（软间隔惩罚因子）
            gamma: 核函数系数（for rbf, poly, sigmoid）
            random_state: 随机种子
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.classes_ = None
        self.classifiers_ = {}  # 存储所有二分类器
        self.class_pairs_ = []  # 存储类别对

    def fit(self, X, y):
        """
        训练模型

        对每一对类别(ci, cj)训练一个二分类SVM:
        优化目标: min 1/2||w||² + C·Σξi
        约束条件: yi(w·xi + b) ≥ 1 - ξi, ξi ≥ 0

        Args:
            X: 训练特征，shape (n_samples, n_features)
            y: 训练标签，shape (n_samples,)
        """
        print("\n" + "=" * 80)
        print("训练支持向量机分类器 (OVO策略)")
        print("=" * 80)

        X = np.array(X)
        y = np.array(y)

        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        print(f"\n训练集大小: {X.shape[0]} 样本")
        print(f"特征维度: {X.shape[1]}")
        print(f"类别数量: {n_classes}")
        print(f"核函数: {self.kernel}")
        print(f"正则化参数 C: {self.C}")

        # 生成所有类别对 (ci, cj)
        self.class_pairs_ = list(combinations(self.classes_, 2))
        n_classifiers = len(self.class_pairs_)

        print(f"\nOVO策略: 需要训练 {n_classifiers} 个二分类器")
        print(f"  公式: K(K-1)/2 = {n_classes}×{n_classes-1}/2 = {n_classifiers}")

        # 训练每个二分类器
        print("\n开始训练二分类器...")
        start_time = time.time()

        for idx, (class_i, class_j) in enumerate(self.class_pairs_, 1):
            # 获取类别ci和cj的样本
            mask = (y == class_i) | (y == class_j)
            X_pair = X[mask]
            y_pair = y[mask]

            # 将标签转换为+1和-1
            y_binary = np.where(y_pair == class_i, 1, -1)

            # 训练二分类SVM
            clf = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                random_state=self.random_state
            )
            clf.fit(X_pair, y_binary)

            # 保存分类器
            self.classifiers_[(class_i, class_j)] = clf

            if idx % 2 == 0 or idx == n_classifiers:
                print(f"  进度: {idx}/{n_classifiers} 完成")

        train_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {train_time:.2f}秒")

        return self

    def predict(self, X):
        """
        预测类别

        对每个测试样本:
        1. 对所有K(K-1)/2个二分类器进行预测
        2. 每个分类器投票给对应的类别
        3. 选择票数最高的类别

        投票规则:
        若 fij(x) = +1, 则 Votes(ci) += 1
        若 fij(x) = -1, 则 Votes(cj) += 1

        最终: C = argmax_c Votes(c)

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            y_pred: 预测标签，shape (n_samples,)
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # 初始化投票矩阵
        votes = np.zeros((n_samples, n_classes))

        # 对每个二分类器进行预测
        for (class_i, class_j), clf in self.classifiers_.items():
            # 获取二分类预测结果
            predictions = clf.predict(X)

            # 投票
            # predictions = +1 → 投票给 class_i
            # predictions = -1 → 投票给 class_j
            idx_i = np.where(self.classes_ == class_i)[0][0]
            idx_j = np.where(self.classes_ == class_j)[0][0]

            votes[predictions == 1, idx_i] += 1
            votes[predictions == -1, idx_j] += 1

        # 选择票数最多的类别
        y_pred_idx = np.argmax(votes, axis=1)
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
    print("支持向量机分类器实验 (OVO策略)")
    print("=" * 80)

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    # 2. 训练模型
    start_time = time.time()
    model = SVMClassifierOVO(
        kernel='linear',  # 线性核适合高维文本特征
        C=1.0,            # 正则化参数
        random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\n总训练时间: {train_time:.2f}秒")

    # 3. 评估模型
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. 保存模型
    model_path = 'models/svm_classifier.pkl'
    import os
    os.makedirs('models', exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n模型已保存至: {model_path}")

    # 5. 保存结果
    results_path = 'models/svm_results.pkl'
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
