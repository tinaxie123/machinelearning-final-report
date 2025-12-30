"""
THUCNews 文本分类 - 从底层实现
包含：TF-IDF特征构建、SVM(SMO算法)、XGBoost
作者：从零实现，不使用sklearn等机器学习库
"""

import numpy as np
import math
import re
import os
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import random
import pickle

# ============================================================================
# 第一部分：文本预处理与特征构建
# ============================================================================

class ChineseTokenizer:
    """
    简单的中文分词器（基于字符和N-gram）
    对于中文文本分类，字符级别+N-gram特征能取得不错效果
    """
    def __init__(self, max_word_len: int = 5, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_word_len = max_word_len
        self.ngram_range = ngram_range
        self.word_dict = set()
        # 停用词列表
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '那', '什么', '他', '她', '它',
            '们', '这个', '那个', '之', '与', '及', '或', '但', '如果', '因为',
            '所以', '然后', '可以', '这样', '那样', '怎么', '为什么', '什么样',
            '如何', '多少', '几', '哪', '谁', '哪里', '怎样', '吗', '呢', '吧',
            '啊', '呀', '哦', '嗯', '啦', '哈', '嘿', '噢', '唉', '哎',
            '\n', '\t', ' ', '，', '。', '！', '？', '、', '；', '：',
            '"', '"', ''', ''', '（', '）', '【', '】', '《', '》',
        ])
    
    def add_words(self, words: List[str]):
        """添加词典词汇"""
        self.word_dict.update(words)
    
    def _extract_ngrams(self, chars: List[str], n: int) -> List[str]:
        """提取N-gram"""
        ngrams = []
        for i in range(len(chars) - n + 1):
            ngram = ''.join(chars[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def tokenize(self, text: str) -> List[str]:
        """
        基于字符和N-gram的分词
        """
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        tokens = []
        
        # 提取中文字符
        chinese_chars = []
        i = 0
        while i < len(text):
            char = text[i]
            # 先检查中文字符（重要：必须在isalpha之前检查）
            if '\u4e00' <= char <= '\u9fa5':
                chinese_chars.append(char)
                i += 1
            # 英文或数字
            elif char.isalnum():
                # 先处理之前积累的中文字符
                if chinese_chars:
                    for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                        ngrams = self._extract_ngrams(chinese_chars, n)
                        for ng in ngrams:
                            if ng not in self.stopwords:
                                tokens.append(ng)
                    chinese_chars = []
                
                j = i
                while j < len(text) and text[j].isalnum() and not ('\u4e00' <= text[j] <= '\u9fa5'):
                    j += 1
                word = text[i:j].lower()
                if word not in self.stopwords and len(word) > 1:
                    tokens.append(word)
                i = j
            else:
                # 遇到非中文字符，处理之前积累的中文字符
                if chinese_chars:
                    # 提取N-gram
                    for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                        ngrams = self._extract_ngrams(chinese_chars, n)
                        for ng in ngrams:
                            if ng not in self.stopwords and len(ng) >= n:
                                tokens.append(ng)
                    chinese_chars = []
                i += 1
        
        # 处理最后剩余的中文字符
        if chinese_chars:
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams = self._extract_ngrams(chinese_chars, n)
                for ng in ngrams:
                    if ng not in self.stopwords and len(ng) >= n:
                        tokens.append(ng)
        
        return tokens


class TFIDFVectorizer:
    """
    TF-IDF向量化器 - 从底层实现
    TF(t,d) = 词t在文档d中出现的次数 / 文档d的总词数
    IDF(t) = log(总文档数 / (包含词t的文档数 + 1)) + 1
    TF-IDF(t,d) = TF(t,d) * IDF(t)
    """
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95):
        """
        参数:
            max_features: 最大特征数
            min_df: 最小文档频率（绝对数）
            max_df: 最大文档频率（比例）
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}  # 词汇表 {word: index}
        self.idf_ = None       # IDF值
        self.tokenizer = ChineseTokenizer()
    
    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        学习词汇表和IDF值
        """
        n_docs = len(documents)
        
        # 统计词频和文档频率
        word_doc_freq = Counter()  # 每个词出现在多少文档中
        word_total_freq = Counter()  # 每个词的总出现次数
        
        print("正在分词并统计词频...")
        tokenized_docs = []
        for i, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            tokenized_docs.append(tokens)
            word_total_freq.update(tokens)
            word_doc_freq.update(set(tokens))  # 每个文档只计数一次
            
            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i+1}/{n_docs} 文档")
        
        # 过滤词汇
        max_doc_freq = int(self.max_df * n_docs) if isinstance(self.max_df, float) else self.max_df
        
        filtered_words = []
        for word, doc_freq in word_doc_freq.items():
            if self.min_df <= doc_freq <= max_doc_freq:
                filtered_words.append((word, word_total_freq[word]))
        
        # 按词频排序，取top max_features
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        filtered_words = filtered_words[:self.max_features]
        
        # 构建词汇表
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(filtered_words)}
        
        print(f"词汇表大小: {len(self.vocabulary_)}")
        
        # 计算IDF
        self.idf_ = np.zeros(len(self.vocabulary_))
        for word, idx in self.vocabulary_.items():
            df = word_doc_freq[word]
            self.idf_[idx] = math.log(n_docs / (df + 1)) + 1
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        将文档转换为TF-IDF矩阵
        """
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet!")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        
        # 使用稀疏表示会更高效，但为了简单起见这里用密集矩阵
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        print("正在计算TF-IDF...")
        for i, doc in enumerate(documents):
            tokens = self.tokenizer.tokenize(doc)
            token_counts = Counter(tokens)
            doc_len = len(tokens) if tokens else 1
            
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    tf = count / doc_len
                    tfidf_matrix[i, idx] = tf * self.idf_[idx]
            
            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i+1}/{n_docs} 文档")
        
        # L2归一化
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        tfidf_matrix = tfidf_matrix / norms
        
        return tfidf_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        fit和transform合并
        """
        self.fit(documents)
        return self.transform(documents)


# ============================================================================
# 第二部分：SVM实现（SMO算法）
# ============================================================================

class SVMClassifier:
    """
    支持向量机分类器 - 使用SMO(Sequential Minimal Optimization)算法
    实现二分类SVM，多分类使用OvR(One-vs-Rest)策略
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', 
                 gamma: float = 0.1, tol: float = 1e-3, 
                 max_iter: int = 1000):
        """
        参数:
            C: 正则化参数
            kernel: 核函数类型 ('linear', 'rbf', 'poly')
            gamma: RBF核的参数
            tol: 收敛容忍度
            max_iter: 最大迭代次数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        
        # 模型参数
        self.alpha = None      # 拉格朗日乘子
        self.b = 0             # 偏置
        self.X = None          # 训练数据
        self.y = None          # 训练标签
        self.support_vectors_ = None
        
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        计算核函数值
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        预计算核矩阵以加速训练
        """
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._kernel_function(X[i], X[j])
                K[j, i] = K[i, j]
        return K
    
    def _predict_one(self, x: np.ndarray) -> float:
        """
        对单个样本预测（返回决策值）
        """
        result = self.b
        for i in range(len(self.alpha)):
            if self.alpha[i] > 1e-8:  # 只考虑支持向量
                result += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
        return result
    
    def _smo_step(self, i: int, K: np.ndarray) -> int:
        """
        SMO算法的一步优化
        选择第二个变量并优化alpha_i和alpha_j
        """
        y = self.y
        alpha = self.alpha
        n = len(y)
        
        # 计算误差
        E_i = sum(alpha[k] * y[k] * K[i, k] for k in range(n)) + self.b - y[i]
        
        # 检查KKT条件是否被违反
        if not ((y[i] * E_i < -self.tol and alpha[i] < self.C) or
                (y[i] * E_i > self.tol and alpha[i] > 0)):
            return 0
        
        # 选择第二个变量j（使用启发式方法）
        j = self._select_j(i, E_i, K)
        if j == -1:
            return 0
        
        E_j = sum(alpha[k] * y[k] * K[j, k] for k in range(n)) + self.b - y[j]
        
        # 保存旧值
        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]
        
        # 计算边界L和H
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(self.C, self.C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - self.C)
            H = min(self.C, alpha[i] + alpha[j])
        
        if L >= H:
            return 0
        
        # 计算eta
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        if eta >= 0:
            return 0
        
        # 更新alpha_j
        alpha[j] = alpha_j_old - y[j] * (E_i - E_j) / eta
        
        # 裁剪alpha_j
        if alpha[j] > H:
            alpha[j] = H
        elif alpha[j] < L:
            alpha[j] = L
        
        # 检查变化是否足够大
        if abs(alpha[j] - alpha_j_old) < 1e-5:
            return 0
        
        # 更新alpha_i
        alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
        
        # 更新偏置b
        b1 = self.b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - \
             y[j] * (alpha[j] - alpha_j_old) * K[i, j]
        b2 = self.b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - \
             y[j] * (alpha[j] - alpha_j_old) * K[j, j]
        
        if 0 < alpha[i] < self.C:
            self.b = b1
        elif 0 < alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        return 1
    
    def _select_j(self, i: int, E_i: float, K: np.ndarray) -> int:
        """
        启发式选择第二个变量
        选择使|E_i - E_j|最大的j
        """
        n = len(self.y)
        max_delta = 0
        j = -1
        
        # 在非边界样本中寻找
        non_bound = [k for k in range(n) if 0 < self.alpha[k] < self.C]
        
        candidates = non_bound if non_bound else range(n)
        
        for k in candidates:
            if k == i:
                continue
            E_k = sum(self.alpha[m] * self.y[m] * K[k, m] for m in range(n)) + self.b - self.y[k]
            delta = abs(E_i - E_k)
            if delta > max_delta:
                max_delta = delta
                j = k
        
        # 如果没找到，随机选择
        if j == -1:
            j = i
            while j == i:
                j = random.randint(0, n - 1)
        
        return j
    
    def fit_binary(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        训练二分类SVM
        """
        n_samples = X.shape[0]
        
        # 初始化
        self.X = X
        self.y = y.astype(float)
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # 预计算核矩阵
        print("  计算核矩阵...")
        K = self._compute_kernel_matrix(X)
        
        # SMO主循环
        print("  开始SMO优化...")
        passes = 0
        iteration = 0
        
        while passes < 5 and iteration < self.max_iter:
            num_changed = 0
            
            for i in range(n_samples):
                num_changed += self._smo_step(i, K)
            
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
            
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"    迭代 {iteration}, 变化数: {num_changed}")
        
        # 记录支持向量
        self.support_vectors_ = X[self.alpha > 1e-8]
        print(f"  支持向量数量: {len(self.support_vectors_)}")
        
        return self
    
    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        二分类预测
        """
        predictions = np.array([self._predict_one(x) for x in X])
        return np.sign(predictions)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        返回决策值
        """
        return np.array([self._predict_one(x) for x in X])


class MultiClassSVM:
    """
    多分类SVM - 使用One-vs-Rest策略
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', 
                 gamma: float = 0.1, tol: float = 1e-3, max_iter: int = 500):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.classifiers = {}
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassSVM':
        """
        训练多分类SVM
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        print(f"训练 {n_classes} 个二分类器 (One-vs-Rest)")
        
        for i, cls in enumerate(self.classes_):
            print(f"\n训练分类器 {i+1}/{n_classes} (类别: {cls})")
            
            # 构建二分类标签
            y_binary = np.where(y == cls, 1, -1)
            
            # 训练二分类器
            clf = SVMClassifier(
                C=self.C, kernel=self.kernel, gamma=self.gamma,
                tol=self.tol, max_iter=self.max_iter
            )
            clf.fit_binary(X, y_binary)
            self.classifiers[cls] = clf
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        # 获取每个分类器的决策值
        decision_values = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, cls in enumerate(self.classes_):
            decision_values[:, i] = self.classifiers[cls].decision_function(X)
        
        # 选择决策值最大的类别
        predictions = self.classes_[np.argmax(decision_values, axis=1)]
        
        return predictions


# ============================================================================
# 第三部分：XGBoost实现
# ============================================================================

class DecisionTreeNode:
    """
    决策树节点
    """
    def __init__(self):
        self.feature_idx = None    # 分裂特征索引
        self.threshold = None      # 分裂阈值
        self.left = None          # 左子树
        self.right = None         # 右子树
        self.is_leaf = False      # 是否为叶节点
        self.value = 0            # 叶节点的预测值


class XGBoostTree:
    """
    XGBoost的单棵决策树（CART回归树）
    使用二阶泰勒展开的损失函数
    """
    
    def __init__(self, max_depth: int = 6, min_samples_split: int = 2,
                 reg_lambda: float = 1.0, gamma: float = 0.0,
                 min_child_weight: float = 1.0):
        """
        参数:
            max_depth: 树的最大深度
            min_samples_split: 节点分裂所需的最小样本数
            reg_lambda: L2正则化系数
            gamma: 分裂所需的最小增益
            min_child_weight: 子节点所需的最小权重和
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.root = None
    
    def _calc_leaf_value(self, g: np.ndarray, h: np.ndarray) -> float:
        """
        计算叶节点的最优值
        w* = -G / (H + lambda)
        """
        G = np.sum(g)
        H = np.sum(h)
        return -G / (H + self.reg_lambda)
    
    def _calc_split_gain(self, g_left: np.ndarray, h_left: np.ndarray,
                         g_right: np.ndarray, h_right: np.ndarray) -> float:
        """
        计算分裂增益
        Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
        """
        G_L, H_L = np.sum(g_left), np.sum(h_left)
        G_R, H_R = np.sum(g_right), np.sum(h_right)
        
        # 检查最小权重约束
        if H_L < self.min_child_weight or H_R < self.min_child_weight:
            return -np.inf
        
        gain = 0.5 * (
            G_L ** 2 / (H_L + self.reg_lambda) +
            G_R ** 2 / (H_R + self.reg_lambda) -
            (G_L + G_R) ** 2 / (H_L + H_R + self.reg_lambda)
        ) - self.gamma
        
        return gain
    
    def _find_best_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray,
                         feature_indices: List[int]) -> Tuple[int, float, float]:
        """
        寻找最佳分裂点
        返回: (最佳特征索引, 最佳阈值, 最佳增益)
        """
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        n_samples = X.shape[0]
        
        for feature_idx in feature_indices:
            # 获取该特征的所有唯一值作为候选阈值
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 如果唯一值太多，采样一部分
            if len(unique_values) > 100:
                unique_values = np.percentile(feature_values, np.linspace(0, 100, 100))
            
            for threshold in unique_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                gain = self._calc_split_gain(
                    g[left_mask], h[left_mask],
                    g[right_mask], h[right_mask]
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, g: np.ndarray, h: np.ndarray,
                    depth: int, feature_indices: List[int]) -> DecisionTreeNode:
        """
        递归构建决策树
        """
        node = DecisionTreeNode()
        n_samples = X.shape[0]
        
        # 停止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            node.is_leaf = True
            node.value = self._calc_leaf_value(g, h)
            return node
        
        # 寻找最佳分裂
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, g, h, feature_indices
        )
        
        # 如果没有有效分裂
        if best_gain <= 0:
            node.is_leaf = True
            node.value = self._calc_leaf_value(g, h)
            return node
        
        # 分裂
        node.feature_idx = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.left = self._build_tree(X[left_mask], g[left_mask], h[left_mask],
                                     depth + 1, feature_indices)
        node.right = self._build_tree(X[right_mask], g[right_mask], h[right_mask],
                                      depth + 1, feature_indices)
        
        return node
    
    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray,
            feature_indices: Optional[List[int]] = None):
        """
        根据一阶和二阶梯度拟合树
        """
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        self.root = self._build_tree(X, g, h, 0, feature_indices)
        return self
    
    def _predict_one(self, x: np.ndarray, node: DecisionTreeNode) -> float:
        """
        对单个样本预测
        """
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        """
        return np.array([self._predict_one(x, self.root) for x in X])


class XGBoostClassifier:
    """
    XGBoost分类器 - 从底层实现
    支持二分类和多分类
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, min_samples_split: int = 2,
                 reg_lambda: float = 1.0, gamma: float = 0.0,
                 subsample: float = 1.0, colsample_bytree: float = 1.0,
                 min_child_weight: float = 1.0, random_state: int = 42):
        """
        参数:
            n_estimators: 树的数量
            learning_rate: 学习率
            max_depth: 单棵树的最大深度
            min_samples_split: 节点分裂所需的最小样本数
            reg_lambda: L2正则化系数
            gamma: 分裂所需的最小增益
            subsample: 样本采样比例
            colsample_bytree: 特征采样比例
            min_child_weight: 子节点所需的最小权重和
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        
        self.trees = []  # 存储所有的树
        self.classes_ = None
        self.n_classes_ = None
        
        random.seed(random_state)
        np.random.seed(random_state)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid函数（数值稳定版本）
        """
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax函数
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _binary_gradient_hessian(self, y: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        二分类的梯度和Hessian（使用对数损失）
        损失函数: L = -y*log(p) - (1-y)*log(1-p)
        梯度: g = p - y
        Hessian: h = p * (1 - p)
        """
        p = self._sigmoid(pred)
        g = p - y
        h = p * (1 - p)
        # 防止Hessian为0
        h = np.maximum(h, 1e-6)
        return g, h
    
    def _multiclass_gradient_hessian(self, y_onehot: np.ndarray, 
                                      pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        多分类的梯度和Hessian（使用softmax交叉熵损失）
        """
        p = self._softmax(pred)
        g = p - y_onehot
        h = p * (1 - p)
        h = np.maximum(h, 1e-6)
        return g, h
    
    def _subsample_data(self, X: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        对数据进行采样
        """
        if self.subsample < 1.0:
            n_subsample = int(n_samples * self.subsample)
            indices = np.random.choice(n_samples, n_subsample, replace=False)
            return indices
        else:
            return np.arange(n_samples)
    
    def _subsample_features(self, n_features: int) -> List[int]:
        """
        对特征进行采样
        """
        if self.colsample_bytree < 1.0:
            n_col = int(n_features * self.colsample_bytree)
            feature_indices = list(np.random.choice(n_features, n_col, replace=False))
        else:
            feature_indices = list(range(n_features))
        return feature_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifier':
        """
        训练XGBoost模型
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            # 二分类
            return self._fit_binary(X, y)
        else:
            # 多分类
            return self._fit_multiclass(X, y)
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifier':
        """
        训练二分类模型
        """
        n_samples, n_features = X.shape
        
        # 初始化预测值
        pred = np.zeros(n_samples)
        
        self.trees = []
        
        print("开始训练XGBoost (二分类)...")
        for i in range(self.n_estimators):
            # 计算梯度和Hessian
            g, h = self._binary_gradient_hessian(y, pred)
            
            # 采样
            sample_indices = self._subsample_data(X, n_samples)
            feature_indices = self._subsample_features(n_features)
            
            # 训练一棵树
            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                min_child_weight=self.min_child_weight
            )
            tree.fit(X[sample_indices], g[sample_indices], h[sample_indices], 
                    feature_indices)
            
            # 更新预测
            pred += self.learning_rate * tree.predict(X)
            
            self.trees.append(tree)
            
            if (i + 1) % 10 == 0:
                # 计算当前损失
                p = self._sigmoid(pred)
                p = np.clip(p, 1e-10, 1 - 1e-10)
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                acc = np.mean((pred >= 0).astype(int) == y)
                print(f"  树 {i+1}/{self.n_estimators}, 损失: {loss:.4f}, 准确率: {acc:.4f}")
        
        return self
    
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostClassifier':
        """
        训练多分类模型（One-vs-All风格，每个类别一组树）
        """
        n_samples, n_features = X.shape
        
        # One-hot编码
        y_onehot = np.zeros((n_samples, self.n_classes_))
        for i, cls in enumerate(self.classes_):
            y_onehot[y == cls, i] = 1
        
        # 初始化预测值
        pred = np.zeros((n_samples, self.n_classes_))
        
        # 每个类别的树列表
        self.trees = [[] for _ in range(self.n_classes_)]
        
        print(f"开始训练XGBoost (多分类, {self.n_classes_}个类别)...")
        for i in range(self.n_estimators):
            # 计算梯度和Hessian
            g, h = self._multiclass_gradient_hessian(y_onehot, pred)
            
            # 采样
            sample_indices = self._subsample_data(X, n_samples)
            feature_indices = self._subsample_features(n_features)
            
            # 为每个类别训练一棵树
            for k in range(self.n_classes_):
                tree = XGBoostTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                    min_child_weight=self.min_child_weight
                )
                tree.fit(X[sample_indices], g[sample_indices, k], 
                        h[sample_indices, k], feature_indices)
                
                # 更新预测
                pred[:, k] += self.learning_rate * tree.predict(X)
                
                self.trees[k].append(tree)
            
            if (i + 1) % 10 == 0:
                # 计算当前准确率
                pred_labels = self.classes_[np.argmax(pred, axis=1)]
                acc = np.mean(pred_labels == y)
                print(f"  迭代 {i+1}/{self.n_estimators}, 准确率: {acc:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        """
        if self.n_classes_ == 2:
            pred = np.zeros(X.shape[0])
            for tree in self.trees:
                pred += self.learning_rate * tree.predict(X)
            proba = self._sigmoid(pred)
            return np.column_stack([1 - proba, proba])
        else:
            pred = np.zeros((X.shape[0], self.n_classes_))
            for k in range(self.n_classes_):
                for tree in self.trees[k]:
                    pred[:, k] += self.learning_rate * tree.predict(X)
            return self._softmax(pred)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ============================================================================
# 第四部分：评估指标
# ============================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算准确率"""
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                        average: str = 'macro') -> Tuple[float, float, float]:
    """
    计算精确率、召回率、F1分数
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for cls in classes:
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(np.sum(y_true == cls))
    
    if average == 'macro':
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    elif average == 'weighted':
        total = sum(supports)
        return (
            sum(p * s for p, s in zip(precisions, supports)) / total,
            sum(r * s for r, s in zip(recalls, supports)) / total,
            sum(f * s for f, s in zip(f1s, supports)) / total
        )
    else:
        return precisions, recalls, f1s


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    计算混淆矩阵
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1
    
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                          target_names: List[str] = None) -> str:
    """
    生成分类报告
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [str(cls) for cls in classes]
    
    # 计算每个类别的指标
    precisions, recalls, f1s = precision_recall_f1(y_true, y_pred, average=None)
    
    report = "分类报告\n"
    report += "=" * 60 + "\n"
    report += f"{'类别':<15} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'支持数':<10}\n"
    report += "-" * 60 + "\n"
    
    for i, cls in enumerate(classes):
        support = np.sum(y_true == cls)
        report += f"{target_names[i]:<15} {precisions[i]:<12.4f} {recalls[i]:<12.4f} {f1s[i]:<12.4f} {support:<10}\n"
    
    report += "-" * 60 + "\n"
    
    # 宏平均
    macro_p, macro_r, macro_f1 = precision_recall_f1(y_true, y_pred, average='macro')
    report += f"{'宏平均':<15} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}\n"
    
    # 加权平均
    weighted_p, weighted_r, weighted_f1 = precision_recall_f1(y_true, y_pred, average='weighted')
    report += f"{'加权平均':<15} {weighted_p:<12.4f} {weighted_r:<12.4f} {weighted_f1:<12.4f}\n"
    
    report += "=" * 60 + "\n"
    report += f"准确率: {accuracy_score(y_true, y_pred):.4f}\n"
    
    return report


# ============================================================================
# 第五部分：数据加载与主程序
# ============================================================================

def load_thucnews_data(data_dir: str, categories: List[str] = None, 
                       max_samples_per_class: int = 500) -> Tuple[List[str], List[str]]:
    """
    加载THUCNews数据集
    
    THUCNews数据集结构:
    data_dir/
        类别1/
            文件1.txt
            文件2.txt
            ...
        类别2/
            ...
    """
    texts = []
    labels = []
    
    if categories is None:
        categories = os.listdir(data_dir)
        categories = [c for c in categories if os.path.isdir(os.path.join(data_dir, c))]
    
    print(f"加载类别: {categories}")
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            continue
        
        files = os.listdir(category_dir)
        files = [f for f in files if f.endswith('.txt')]
        
        # 限制每个类别的样本数
        if max_samples_per_class and len(files) > max_samples_per_class:
            files = random.sample(files, max_samples_per_class)
        
        print(f"  {category}: {len(files)} 样本")
        
        for filename in files:
            filepath = os.path.join(category_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(category)
            except Exception as e:
                pass
    
    return texts, labels


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练集和测试集
    """
    np.random.seed(random_state)
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def create_sample_data() -> Tuple[List[str], np.ndarray]:
    """
    创建示例数据（当没有THUCNews数据时使用）
    """
    print("创建示例数据集...")
    
    # 示例文本数据 - 扩展版
    sports_texts = [
        "中国男篮在亚运会上取得了优异成绩姚明表示非常满意球队配合默契",
        "足球世界杯预选赛中国队迎战日本队比赛非常激烈球迷热情高涨",
        "NBA总决赛湖人队战胜凯尔特人队科比获得总决赛MVP表现出色",
        "刘翔在奥运会上打破世界纪录成为中国田径的骄傲创造历史",
        "游泳世锦赛孙杨夺得金牌创造了新的亚洲纪录展现实力",
        "网球公开赛李娜击败对手晋级决赛创造历史中国网球突破",
        "乒乓球世界杯中国队包揽金银铜牌展现统治力实力强劲",
        "羽毛球比赛林丹战胜李宗伟获得冠军精彩对决",
        "篮球比赛火箭队击败勇士队哈登砍下50分表现惊艳",
        "足球联赛恒大队夺冠郑智当选最佳球员实至名归",
        "中超联赛激战正酣各队争夺积分榜首位置竞争激烈",
        "CBA联赛新赛季开始各队实力有所变化球迷期待",
        "奥运会田径比赛中国选手表现出色获得多枚奖牌",
        "世界杯足球赛预选赛中国队客场作战取得平局",
        "亚冠联赛中国球队表现优异晋级淘汰赛阶段",
        "全运会比赛如火如荼各省队伍激烈角逐金牌",
        "排球世界杯中国女排战胜强敌取得重要胜利",
        "马拉松比赛在城市举行数万名选手参加跑步",
        "体操世锦赛中国队获得团体冠军个人项目也有斩获",
        "跳水比赛中国选手完美发挥获得满分夺得金牌",
    ]
    
    tech_texts = [
        "苹果公司发布新款iPhone搭载最新的处理器芯片性能强大",
        "人工智能技术在医疗领域取得重大突破可以辅助诊断疾病",
        "特斯拉推出新款电动汽车续航里程超过500公里科技感十足",
        "华为发布5G技术白皮书引领行业发展方向技术领先",
        "谷歌推出新的搜索算法提升用户搜索体验更加智能",
        "微软发布Windows新版本增加了许多新功能界面优化",
        "阿里巴巴云计算业务增长迅速市场份额扩大技术实力强",
        "百度自动驾驶技术取得进展完成城市道路测试安全可靠",
        "腾讯推出新的社交应用用户数量快速增长功能丰富",
        "小米发布智能手机新品性价比极高配置出色",
        "人工智能深度学习模型在图像识别领域取得突破准确率提升",
        "芯片技术研发取得进展国产芯片性能不断提高",
        "量子计算机研究取得新进展计算能力大幅提升",
        "机器人技术在制造业广泛应用提高生产效率",
        "物联网技术发展迅速智能家居设备普及率提高",
        "区块链技术在金融领域应用探索不断深入",
        "虚拟现实VR技术在娱乐和教育领域应用广泛",
        "无人机技术发展快速在农业和物流领域应用",
        "智能穿戴设备市场增长用户健康管理需求提升",
        "大数据分析技术帮助企业做出更好的商业决策",
    ]
    
    finance_texts = [
        "股市今日大涨上证指数突破3000点投资者信心增强",
        "央行宣布降息利率下调0.25个百分点刺激经济增长",
        "房地产市场调控政策出台限购范围扩大抑制房价",
        "人民币汇率走势稳定外汇储备充足经济基本面良好",
        "银行理财产品收益率下降投资者转向基金寻求更高回报",
        "上市公司季度财报发布业绩超出预期股价上涨",
        "期货市场大宗商品价格波动投资风险加大需谨慎",
        "保险行业发展迅速健康险需求增加市场潜力大",
        "互联网金融监管趋严平台合规压力大整改进行中",
        "债券市场收益率上升国债受到追捧安全性高",
        "股票市场成交量放大资金流入明显投资情绪活跃",
        "基金发行规模创新高投资者认购热情高涨",
        "银行存款利率调整储户收益发生变化理财观念转变",
        "外资持续流入A股市场国际投资者看好中国经济",
        "企业债发行增加融资成本有所下降利好实体经济",
        "私募基金规模扩大高净值投资者配置需求增加",
        "金融科技发展迅速银行数字化转型加速进行",
        "证券交易所推出新规范规范市场行为保护投资者",
        "经济数据发布GDP增长符合预期经济运行平稳",
        "货币政策保持稳健流动性合理充裕支持实体经济",
    ]
    
    entertainment_texts = [
        "电影票房创新高国产电影表现优异观众口碑良好",
        "明星演唱会门票秒光粉丝热情高涨现场气氛火爆",
        "电视剧收视率创纪录剧情引发热议话题度高",
        "综艺节目收视率攀升嘉宾阵容强大节目精彩纷呈",
        "音乐排行榜新歌上榜歌手人气上涨粉丝支持",
        "颁奖典礼明星云集红毯造型亮眼备受关注",
        "新剧开拍演员阵容曝光备受期待粉丝翘首以盼",
        "艺人恋情公开粉丝送上祝福祝愿幸福美满",
        "演员新作品上映演技获得好评观众认可",
        "歌手发布新专辑销量破百万人气火爆",
        "电影节开幕众多影片参展评委阵容强大",
        "电视剧热播收视率持续攀升成为热门话题",
        "明星参加真人秀节目表现真实圈粉无数",
        "音乐综艺节目播出选手实力强劲竞争激烈",
        "演员凭借新作获得最佳表演奖实力认可",
        "电影改编自热门小说原著粉丝期待值高",
        "艺人公益活动献爱心传递正能量获赞",
        "演唱会巡演开启粉丝全国各地追随偶像",
        "电视剧翻拍经典新演员演绎引发讨论",
        "娱乐新闻热搜明星动态引发网友关注讨论",
    ]
    
    education_texts = [
        "高考成绩公布考生查询通道开启学生紧张等待",
        "大学录取分数线公布志愿填报开始选择学校",
        "中小学减负政策出台作业量明显减少学生减压",
        "教师资格考试报名开始考生人数创新高竞争激烈",
        "研究生招生简章发布招生规模扩大机会增多",
        "在线教育平台发展迅速用户数量增加学习便利",
        "学校举办科技创新比赛学生积极参与展示才华",
        "教育改革方案出台素质教育受重视全面发展",
        "留学申请季到来学生准备材料积极备考",
        "职业教育发展政策支持培训机构增多就业导向",
        "高校招生改革推进综合评价录取试点扩大",
        "考研人数持续增长竞争压力加大复习紧张",
        "中小学课程改革新教材投入使用内容更新",
        "教师待遇提升优秀人才加入教育行业发展",
        "学生综合素质评价体系建立全面发展导向",
        "家庭教育重要性凸显家长参与学生成长",
        "校园安全管理加强保护学生身心健康",
        "教育公平推进农村教育资源得到改善",
        "高等教育国际化交流项目增多视野拓宽",
        "终身学习理念普及成人教育需求增长",
    ]
    
    texts = sports_texts + tech_texts + finance_texts + entertainment_texts + education_texts
    labels = ['体育'] * 20 + ['科技'] * 20 + ['财经'] * 20 + ['娱乐'] * 20 + ['教育'] * 20
    
    return texts, np.array(labels)


def main():
    """
    主程序
    """
    print("=" * 70)
    print("THUCNews 文本分类 - 从底层实现 SVM 和 XGBoost")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n1. 加载数据")
    print("-" * 50)
    
    # 尝试加载THUCNews数据，如果不存在则使用示例数据
    thucnews_path = "./THUCNews"  # THUCNews数据集路径
    
    if os.path.exists(thucnews_path):
        categories = ['体育', '科技', '财经', '娱乐', '教育']  # 选择部分类别
        texts, labels = load_thucnews_data(
            thucnews_path, 
            categories=categories,
            max_samples_per_class=200  # 每类最多200个样本
        )
        labels = np.array(labels)
    else:
        print(f"未找到THUCNews数据集，使用示例数据")
        texts, labels = create_sample_data()
    
    print(f"总样本数: {len(texts)}")
    print(f"类别分布: {dict(Counter(labels))}")
    
    # 2. 特征提取
    print("\n2. TF-IDF特征提取")
    print("-" * 50)
    
    vectorizer = TFIDFVectorizer(max_features=5000, min_df=1, max_df=1.0)
    X = vectorizer.fit_transform(texts)
    
    print(f"特征矩阵形状: {X.shape}")
    
    # 3. 划分训练集和测试集
    print("\n3. 划分数据集")
    print("-" * 50)
    
    # 标签编码
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {len(y_train)}")
    print(f"测试集大小: {len(y_test)}")
    
    # 4. 训练SVM
    print("\n4. 训练SVM分类器")
    print("-" * 50)
    
    # 对于大数据集，可以使用较小的子集来训练SVM（SMO算法较慢）
    if len(y_train) > 200:
        print(f"数据集较大，使用前200个样本训练SVM...")
        svm_train_size = 200
        svm_X_train = X_train[:svm_train_size]
        svm_y_train = y_train[:svm_train_size]
    else:
        svm_X_train = X_train
        svm_y_train = y_train
    
    svm_clf = MultiClassSVM(C=1.0, kernel='linear', tol=1e-3, max_iter=200)
    svm_clf.fit(svm_X_train, svm_y_train)
    
    # SVM预测
    svm_pred = svm_clf.predict(X_test)
    
    print("\nSVM分类结果:")
    print(classification_report(y_test, svm_pred, target_names=list(unique_labels)))
    
    # 5. 训练XGBoost
    print("\n5. 训练XGBoost分类器")
    print("-" * 50)
    
    xgb_clf = XGBoostClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=5,
        reg_lambda=1.0,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    
    # XGBoost预测
    xgb_pred = xgb_clf.predict(X_test)
    
    print("\nXGBoost分类结果:")
    print(classification_report(y_test, xgb_pred, target_names=list(unique_labels)))
    
    # 6. 模型对比
    print("\n6. 模型对比")
    print("-" * 50)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"SVM 准确率:     {svm_acc:.4f}")
    print(f"XGBoost 准确率: {xgb_acc:.4f}")
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    
    return {
        'vectorizer': vectorizer,
        'svm': svm_clf,
        'xgboost': xgb_clf,
        'unique_labels': unique_labels,
        'results': {
            'svm_accuracy': svm_acc,
            'xgboost_accuracy': xgb_acc
        }
    }


if __name__ == "__main__":
    results = main()
