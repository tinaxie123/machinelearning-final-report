"""
特征提取模块
包含TF-IDF、Word2Vec、BERT三种特征提取器
"""

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# 延迟导入gensim和transformers，避免启动时的兼容性问题
# from gensim.models import Word2Vec
# from transformers import BertTokenizer, BertModel
# import torch
from tqdm import tqdm


class TFIDFExtractor:
    """TF-IDF特征提取器"""

    def __init__(self, max_features=8000, min_df=5, max_df=1.0, ngram_range=(1, 2)):
        """
        初始化TF-IDF提取器

        Args:
            max_features: 最大特征数
            min_df: 最小文档频率
            max_df: 最大文档频率
            ngram_range: n-gram范围
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.vectorizer = None

    def fit(self, X_tokens):
        """
        训练TF-IDF模型

        Args:
            X_tokens: 分词列表的列表
        """
        # 将分词列表转换为空格分隔的字符串
        X_text = [' '.join(tokens) for tokens in X_tokens]

        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            token_pattern=r"(?u)\b\w+\b"
        )

        # 训练
        self.vectorizer.fit(X_text)

        print(f"[OK] TF-IDF模型训练完成")
        print(f"  词汇表大小: {len(self.vectorizer.vocabulary_)}")

    def transform(self, X_tokens):
        """
        转换为TF-IDF特征

        Args:
            X_tokens: 分词列表的列表

        Returns:
            TF-IDF特征矩阵 (n_samples, n_features)
        """
        if self.vectorizer is None:
            raise ValueError("模型未训练，请先调用fit()")

        X_text = [' '.join(tokens) for tokens in X_tokens]
        X_tfidf = self.vectorizer.transform(X_text).toarray()

        return X_tfidf

    def fit_transform(self, X_tokens):
        """
        训练并转换

        Args:
            X_tokens: 分词列表的列表

        Returns:
            TF-IDF特征矩阵
        """
        self.fit(X_tokens)
        return self.transform(X_tokens)

    def save(self, path):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"[OK] TF-IDF模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"[OK] TF-IDF模型已加载: {path}")


class Word2VecExtractor:
    """Word2Vec特征提取器（使用TF-IDF加权平均池化）"""

    def __init__(self, vector_size=100, window=5, min_count=5, epochs=10, sg=1, workers=4, learning_rate=0.025):
        """
        初始化Word2Vec提取器

        Args:
            vector_size: 词向量维度
            window: 窗口大小
            min_count: 最小词频
            epochs: 训练轮数
            sg: 0=CBOW, 1=Skip-gram
            workers: 并行训练的线程数
            learning_rate: 学习率
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.workers = workers
        self.learning_rate = learning_rate
        self.model = None
        self.tfidf_weights = None

    def fit(self, X_tokens, tfidf_vectorizer=None):
        """
        训练Word2Vec模型

        Args:
            X_tokens: 分词列表的列表
            tfidf_vectorizer: 已训练的TF-IDF向量化器（用于计算词权重）
        """
        # 延迟导入gensim
        from gensim.models import Word2Vec

        # 训练Word2Vec
        self.model = Word2Vec(
            sentences=X_tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            workers=self.workers
        )

        print(f"[OK] Word2Vec模型训练完成")
        print(f"  词汇量: {len(self.model.wv)}")

        # 计算TF-IDF权重（用于加权平均）
        if tfidf_vectorizer is not None:
            self.tfidf_weights = {}
            vocab = tfidf_vectorizer.vocabulary_
            idf = tfidf_vectorizer.idf_

            for word, idx in vocab.items():
                self.tfidf_weights[word] = idf[idx]

    def transform(self, X_tokens):
        """
        转换为Word2Vec特征（TF-IDF加权平均）

        Args:
            X_tokens: 分词列表的列表

        Returns:
            Word2Vec特征矩阵 (n_samples, vector_size)
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()")

        X_w2v = []
        for tokens in X_tokens:
            # 获取所有词向量和权重
            vectors = []
            weights = []

            for word in tokens:
                if word in self.model.wv:
                    vectors.append(self.model.wv[word])
                    # 使用TF-IDF权重（如果有）
                    weight = self.tfidf_weights.get(word, 1.0) if self.tfidf_weights else 1.0
                    weights.append(weight)

            # 加权平均
            if len(vectors) > 0:
                vectors = np.array(vectors)
                weights = np.array(weights)
                weights = weights / weights.sum()  # 归一化
                sentence_vector = np.average(vectors, axis=0, weights=weights)
            else:
                # 如果没有词在词汇表中，返回零向量
                sentence_vector = np.zeros(self.vector_size)

            X_w2v.append(sentence_vector)

        return np.array(X_w2v)

    def save(self, path):
        """保存模型"""
        self.model.save(path)
        # 同时保存TF-IDF权重
        weights_path = path + '.tfidf_weights.pkl'
        with open(weights_path, 'wb') as f:
            pickle.dump(self.tfidf_weights, f)
        print(f"[OK] Word2Vec模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(path)
        # 加载TF-IDF权重
        weights_path = path + '.tfidf_weights.pkl'
        try:
            with open(weights_path, 'rb') as f:
                self.tfidf_weights = pickle.load(f)
        except:
            self.tfidf_weights = None
        print(f"[OK] Word2Vec模型已加载: {path}")


class BERTExtractor:
    """BERT特征提取器"""

    def __init__(self, model_name='bert-base-chinese', max_length=128, batch_size=32):
        """
        初始化BERT提取器

        Args:
            model_name: BERT模型名称
            max_length: 最大序列长度
            batch_size: 批次大小
        """
        # 延迟导入transformers和torch
        from transformers import BertTokenizer, BertModel
        import torch

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        # 加载BERT模型和分词器
        print(f"[加载] BERT模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

        # 设置为评估模式
        self.model.eval()

        # 使用GPU（如果可用）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"[OK] BERT模型加载完成，设备: {self.device}")

    def transform(self, X_tokens):
        """
        转换为BERT特征

        Args:
            X_tokens: 分词列表的列表

        Returns:
            BERT特征矩阵 (n_samples, 768)
        """
        import torch

        X_bert = []

        # 批量处理
        for i in tqdm(range(0, len(X_tokens), self.batch_size), desc="  提取BERT特征"):
            batch_tokens = X_tokens[i:i + self.batch_size]

            # 将分词列表转换为字符串
            batch_texts = [' '.join(tokens) for tokens in batch_tokens]

            # BERT编码
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # 转移到设备
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # 前向传播（不计算梯度）
            with torch.no_grad():
                outputs = self.model(**encoded)

            # 提取[CLS]对应的向量
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            X_bert.append(cls_embeddings)

        X_bert = np.vstack(X_bert)
        return X_bert
