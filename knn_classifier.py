"""
K近邻分类器 (K-Nearest Neighbors with Cosine Similarity)
使用余弦相似度衡量距离，适合高维稀疏特征
"""

import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time


class KNNClassifier:
    """
    K近邻分类器

    使用余弦相似度作为距离度量:
    sim(x, xi) = (x · xi) / (||x|| ||xi||)

    预测规则:
    y = argmax_c Σ I(yi = c) for xi ∈ NK(x)
    """

    def __init__(self, n_neighbors=5):
        """
        初始化KNN分类器

        Args:
            n_neighbors: 邻居数量K
        """
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        训练模型（存储训练数据）

        KNN是懒惰学习算法，训练阶段只需存储数据

        Args:
            X: 训练特征，shape (n_samples, n_features)
            y: 训练标签，shape (n_samples,)
        """
        print("\n" + "=" * 80)
        print("训练K近邻分类器")
        print("=" * 80)

        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(y)

        print(f"\n训练集大小: {self.X_train_.shape[0]} 样本")
        print(f"特征维度: {self.X_train_.shape[1]}")
        print(f"类别数量: {len(self.classes_)}")
        print(f"邻居数量 K: {self.n_neighbors}")
        print(f"距离度量: 余弦相似度")

        print("\nKNN训练完成（存储训练数据）")

        return self

    def cosine_similarity(self, X_test):
        """
        计算余弦相似度

        sim(x_test, x_train) = (x_test · x_train) / (||x_test|| ||x_train||)

        Args:
            X_test: 测试样本，shape (n_test, n_features)

        Returns:
            similarities: 相似度矩阵，shape (n_test, n_train)
        """
        # 归一化测试样本
        X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10)

        # 归一化训练样本
        X_train_norm = self.X_train_ / (np.linalg.norm(self.X_train_, axis=1, keepdims=True) + 1e-10)

        # 计算余弦相似度 (内积)
        similarities = X_test_norm @ X_train_norm.T

        return similarities

    def predict(self, X):
        """
        预测类别

        对每个测试样本:
        1. 计算与所有训练样本的余弦相似度
        2. 选择相似度最高的K个邻居
        3. 通过投票确定类别

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            y_pred: 预测标签，shape (n_samples,)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # 计算余弦相似度
        similarities = self.cosine_similarity(X)

        # 找到K个最近邻居（相似度最高）
        # argsort默认升序，所以取[-k:]获取最大的K个
        neighbor_indices = np.argsort(similarities, axis=1)[:, -self.n_neighbors:]

        # 获取邻居的标签
        neighbor_labels = self.y_train_[neighbor_indices]

        # 投票选择最频繁的类别
        y_pred = []
        for labels in neighbor_labels:
            # 统计每个类别的出现次数
            unique, counts = np.unique(labels, return_counts=True)
            # 选择出现次数最多的类别
            y_pred.append(unique[np.argmax(counts)])

        y_pred = np.array(y_pred)

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
    print("K近邻分类器实验")
    print("=" * 80)

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    # 2. 训练模型
    start_time = time.time()
    model = KNNClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\n总训练时间: {train_time:.2f}秒")

    # 3. 评估模型
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. 保存模型
    model_path = 'models/knn_classifier.pkl'
    import os
    os.makedirs('models', exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n模型已保存至: {model_path}")

    # 5. 保存结果
    results_path = 'models/knn_results.pkl'
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
