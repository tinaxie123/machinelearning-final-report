"""
逻辑回归分类器 (Logistic Regression with Softmax) - 底层实现
使用梯度下降优化交叉熵损失函数
"""

import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time


class LogisticRegressionClassifier:
    """
    逻辑回归分类器（Softmax多分类）- 底层实现

    多分类概率:
    P(y=k|x) = exp(wk^T x + bk) / Σj exp(wj^T x + bj)

    损失函数（交叉熵）:
    L = -1/N Σi Σk I(yi=k) log P(y=k|xi) + λ/2 ||W||²

    梯度:
    ∂L/∂wk = 1/N Σi (P(y=k|xi) - I(yi=k)) xi + λ wk
    ∂L/∂bk = 1/N Σi (P(y=k|xi) - I(yi=k))

    更新规则:
    wk ← wk - η * ∂L/∂wk
    bk ← bk - η * ∂L/∂bk
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, reg_lambda=0.01,
                 tol=1e-4, random_state=42):
        """
        初始化逻辑回归分类器

        Args:
            learning_rate: 学习率η
            max_iter: 最大迭代次数
            reg_lambda: L2正则化系数λ
            tol: 收敛容忍度
            random_state: 随机种子
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.random_state = random_state
        self.W_ = None  # 权重矩阵 shape (n_features, n_classes)
        self.b_ = None  # 偏置向量 shape (n_classes,)
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.loss_history_ = []

    def softmax(self, z):
        """
        Softmax函数
        P(y=k|x) = exp(zk) / Σj exp(zj)

        使用数值稳定的实现（减去最大值）
        """
        # 减去每行的最大值，防止溢出
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, X, y_onehot):
        """
        计算交叉熵损失

        L = -1/N Σi Σk yk log P(y=k|xi) + λ/2 ||W||²
        """
        n_samples = X.shape[0]

        # 计算预测概率
        z = X @ self.W_ + self.b_
        proba = self.softmax(z)

        # 交叉熵损失
        cross_entropy = -np.sum(y_onehot * np.log(proba + 1e-10)) / n_samples

        # L2正则化
        reg_loss = 0.5 * self.reg_lambda * np.sum(self.W_ ** 2)

        total_loss = cross_entropy + reg_loss

        return total_loss

    def fit(self, X, y):
        """
        训练模型（梯度下降）

        Args:
            X: 训练特征，shape (n_samples, n_features)
            y: 训练标签，shape (n_samples,)
        """
        print("\n" + "=" * 80)
        print("训练逻辑回归分类器 (Softmax) - 底层实现")
        print("=" * 80)

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # 获取类别
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        print(f"\n训练集大小: {n_samples} 样本")
        print(f"特征维度: {n_features}")
        print(f"类别数量: {self.n_classes_}")
        print(f"\n优化参数:")
        print(f"  学习率: {self.learning_rate}")
        print(f"  L2正则化: {self.reg_lambda}")
        print(f"  最大迭代次数: {self.max_iter}")
        print(f"  收敛容忍度: {self.tol}")

        print("\n算法: 批量梯度下降 (Batch Gradient Descent)")
        print("优化目标: 交叉熵损失 + L2正则化")

        # 将标签转换为one-hot编码
        y_onehot = np.zeros((n_samples, self.n_classes_))
        for i, label in enumerate(y):
            class_idx = np.where(self.classes_ == label)[0][0]
            y_onehot[i, class_idx] = 1

        # 初始化权重和偏置
        np.random.seed(self.random_state)
        self.W_ = np.random.randn(n_features, self.n_classes_) * 0.01
        self.b_ = np.zeros(self.n_classes_)

        # 梯度下降
        print("\n开始训练...")
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # 前向传播
            z = X @ self.W_ + self.b_  # shape (n_samples, n_classes)
            proba = self.softmax(z)

            # 计算损失
            loss = self.compute_loss(X, y_onehot)
            self.loss_history_.append(loss)

            # 计算梯度
            # ∂L/∂W = 1/N X^T (P - Y) + λW
            # ∂L/∂b = 1/N Σ(P - Y)
            error = proba - y_onehot  # shape (n_samples, n_classes)

            grad_W = (X.T @ error) / n_samples + self.reg_lambda * self.W_
            grad_b = np.sum(error, axis=0) / n_samples

            # 更新参数
            self.W_ -= self.learning_rate * grad_W
            self.b_ -= self.learning_rate * grad_b

            # 打印进度
            if (iteration + 1) % 100 == 0 or iteration == 0:
                print(f"  迭代 {iteration+1}/{self.max_iter}, 损失: {loss:.6f}")

            # 检查收敛
            if abs(prev_loss - loss) < self.tol:
                print(f"\n在第 {iteration+1} 次迭代时收敛！")
                break

            prev_loss = loss

        print(f"\n训练完成！")
        print(f"最终损失: {self.loss_history_[-1]:.6f}")

        return self

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            proba: 概率矩阵，shape (n_samples, n_classes)
        """
        X = np.array(X, dtype=np.float64)
        z = X @ self.W_ + self.b_
        proba = self.softmax(z)
        return proba

    def predict(self, X):
        """
        预测类别

        Args:
            X: 测试特征，shape (n_samples, n_features)

        Returns:
            y_pred: 预测标签，shape (n_samples,)
        """
        proba = self.predict_proba(X)
        y_pred_idx = np.argmax(proba, axis=1)
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
    print("逻辑回归分类器实验 (Softmax) - 底层实现")
    print("=" * 80)

    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_fused_features()

    # 2. 训练模型
    start_time = time.time()
    model = LogisticRegressionClassifier(
        learning_rate=0.1,  # 学习率
        max_iter=1000,      # 最大迭代次数
        reg_lambda=0.01,    # L2正则化
        tol=1e-4,           # 收敛容忍度
        random_state=42
    )
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\n总训练时间: {train_time:.2f}秒")

    # 3. 评估模型
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. 保存模型
    model_path = 'models/lr_classifier_scratch.pkl'
    import os
    os.makedirs('models', exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n模型已保存至: {model_path}")

    # 5. 保存结果
    results_path = 'models/lr_results_scratch.pkl'
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
