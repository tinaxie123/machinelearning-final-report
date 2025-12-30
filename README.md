# 基于多基模型集成学习的中文新闻文本分类系统

## 项目概述

本项目实现了一个基于多基模型集成学习的中文新闻文本分类系统，采用异构特征融合与双重集成策略，实现高精度的文本分类。

**🎯 快速开始:**
- 如果您已经准备好数据集，请查看 [快速启动指南 (QUICKSTART.md)](QUICKSTART.md)
- 如果要了解特征提取流程，请查看 [特征提取指南 (FEATURE_EXTRACTION_GUIDE.md)](FEATURE_EXTRACTION_GUIDE.md)

## 方法论

### 1. 整体架构

```
数据输入 → 预处理 → 异构特征提取 → 特征融合 → 基模型训练 → 双重集成 → 分类结果
```

### 2. 数据预处理

- **文本清洗**：去除特殊符号、数字、冗余空格
- **中文分词**：使用 jieba 分词工具，加载专业词典
- **停用词过滤**：使用停用词表（749个停用词）
- **编码统一**：UTF-8 编码
- **标准化**：小写转换

**预处理结果统计**：
```
训练集: 10,000样本，平均词数: 8.68
验证集:  5,000样本，平均词数: 8.65
测试集:  5,000样本，平均词数: 8.69

类别分布（每类样本数）:
- 娱乐: 2,000
- 财经: 2,000
- 教育: 2,000
- 体育: 2,000
- 科技: 2,000

保存位置: features/preprocessed/preprocessed_data.pkl
```

### 3. 异构特征提取

#### 3.1 TF-IDF 特征（表层统计特征）

- **词频（TF）**：`TF(t,d) = count(t,d) / len(d)`
- **逆文档频率（IDF）**：`IDF(t) = log((N+1)/(doc_count(t)+1)) + 1`
- **TF-IDF 值**：`TF-IDF(t,d) = TF(t,d) × IDF(t)`
- **特征维度**：8000维（通过 min_df 控制）

#### 3.2 Word2Vec 特征（浅层语义特征）

- **模型架构**：Skip-gram
- **训练参数**：
  - 向量维度：100
  - 窗口大小：5
  - 最小词频：5
  - 迭代次数：10
- **句子向量**：加权平均池化（以 TF-IDF 为权重）

#### 3.3 BERT 特征（深层语义特征）

- **预训练模型**：bert-base-chinese
- **模型参数**：12层 Transformer，768维隐藏层
- **输入格式**：`[CLS] + 分词序列 + [SEP]`
- **最大序列长度**：128
- **特征提取**：提取 [CLS] 对应的768维向量

### 4. 异构特征融合

- **标准化**：Z-score 标准化（均值为0，方差为1）
  ```
  V'_i = (V_i - μ_i) / σ_i
  ```

- **加权融合**：
  ```
  V_fusion = w1 × V'_TF-IDF + w2 × V'_Word2Vec + w3 × V'_BERT
  ```
  - 最优权重：TF-IDF (0.2)、Word2Vec (0.3)、BERT (0.5)
  - 融合后维度：8000维（通过零填充统一）

### 5. 基层模型（8种分类器）

| 模型 | 类型 | 核心优势 |
|------|------|----------|
| 多项式朴素贝叶斯 | 统计学习 | 适合文本计数特征 |
| 支持向量机 (SVM) | 核方法 | 高维数据泛化能力强 |
| K近邻 (KNN) | Instance-based | 适应局部分布模式 |
| 逻辑回归 (LR) | 统计学习 | 高维稀疏数据稳定 |
| 随机森林 (RF) | 集成树 | Bagging，降低过拟合 |
| GBDT | 梯度提升 | 迭代优化残差 |
| XGBoost | 梯度提升 | 正则化，高效特征筛选 |
| LightGBM | 梯度提升 | 快速训练，按叶子分裂 |

### 6. 双重集成融合策略

#### 6.1 加权软投票

- **权重计算**：基于验证集宏平均 F1 值自适应分配
  ```
  w_i = F1_i / Σ(F1_j)
  ```

- **加权融合**：
  ```
  P_vote(c) = Σ(w_i × P_i(c))
  ```

#### 6.2 Stacking 集成

- **元特征构建**：采用 5折 Out-of-Fold (OOF) 交叉验证
  - 避免过拟合
  - 每个样本的元特征来自未见过该样本的基模型

- **元学习器**：带 L2 正则化的 Logistic Regression
  - 正则化参数：C=1.0
  - 惩罚项：L2

- **最终预测**：元学习器输出类别概率分布

### 7. 评估指标

- **宏平均精确率（Macro-P）**：各类别精确率的算术平均
- **宏平均召回率（Macro-R）**：各类别召回率的算术平均
- **宏平均 F1 值（Macro-F1）**：
  ```
  Macro-F1 = 2 × Macro-P × Macro-R / (Macro-P + Macro-R)
  ```
- **准确率（Accuracy）**：整体预测正确的样本占比

### 8. 数据集

- **来源**：THUCNews 数据集
- **类别**：娱乐、财经、教育、体育、科技（5类）
- **样本数量**：10,000条（每类2,000条）
- **划分比例**：训练集:验证集:测试集 = 7:2:1

## 预期性能

| 指标 | 预期值 |
|------|--------|
| 宏平均 F1 值 | 0.91 |
| 宏平均精确率 | 0.91 |
| 宏平均召回率 | 0.89 |
| 准确率 | 0.92 |

## 项目结构

```
mlfinalreport/
├── README.md                   # 项目说明文档
├── requirements.txt            # 依赖包列表
├── config.py                   # 配置文件
├── preprocessing.py            # 数据预处理模块
├── feature_extraction.py       # 特征提取模块
├── feature_fusion.py           # 特征融合模块
├── base_models.py              # 基模型定义
├── ensemble.py                 # 集成学习模块
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
└── main.py                     # 主程序入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

#### 2.1 下载 THUCNews 数据集

1. 访问清华大学自然语言处理实验室官网：http://thuctc.thunlp.org/
2. 下载 THUCNews 数据集（约2GB）
3. 解压到项目的 `data/` 目录

**数据集结构**：
```
data/
├── 财经/
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
├── 教育/
├── 体育/
├── 科技/
├── 娱乐/
└── ...（其他类别）
```

详细说明请参考 [**数据集准备指南 (DATASET_GUIDE.md)**](DATASET_GUIDE.md)

#### 2.2 检查数据集

```bash
# 检查数据集是否正确放置
python prepare_data.py --action check --data-dir data/
```

#### 2.3 创建测试子集（可选）

如果想快速测试，可以创建小规模子集：

```bash
# 每个类别提取1000个样本用于测试
python prepare_data.py --action subset --data-dir data/ --num-samples 1000
# 子集将保存在 data_sample/ 目录
```

### 3. 训练模型

```bash
python main.py --mode train
```

### 4. 评估模型

```bash
python main.py --mode evaluate
```

### 5. 预测

```bash
python main.py --mode predict --text "你的中文新闻文本"
```

## 技术栈

- Python 3.8+
- scikit-learn：传统机器学习模型
- jieba：中文分词
- gensim：Word2Vec 训练
- transformers：BERT 模型
- xgboost、lightgbm：梯度提升树
- numpy、pandas：数据处理

## 核心创新点

1. **异构特征融合**：整合 TF-IDF、Word2Vec、BERT 三种不同层次的特征表示
2. **双重集成策略**：结合加权软投票与 Stacking，充分发挥基模型互补优势
3. **自适应权重分配**：基于验证集 F1 值动态计算基模型权重
4. **OOF 元特征构建**：通过交叉验证避免 Stacking 过拟合

## 参考文献

- 本项目基于机器学习与自然语言处理的经典理论实现
- 数据集：THUCNews 中文新闻文本分类数据集
