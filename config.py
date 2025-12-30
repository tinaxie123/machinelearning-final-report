import os
DATA_CONFIG = {
    # THUCNews数据路径
    'data_path': 'THUCNews/data/',
    'train_file': 'train.txt',
    'dev_file': 'dev.txt',
    'test_file': 'test.txt',
    'class_file': 'class.txt',

    # 类别映射（THUCNews标准：类别编号 -> 类别名）
    'class_mapping': {
        0: 'finance',      # 财经
        1: 'realty',       # 房产
        2: 'stocks',       # 股票
        3: 'education',    # 教育
        4: 'science',      # 科技
        5: 'society',      # 社会
        6: 'politics',     # 时政
        7: 'sports',       # 体育
        8: 'game',         # 游戏
        9: 'entertainment' # 娱乐
    },

    # 中文类别名映射
    'class_mapping_cn': {
        'finance': '财经',
        'realty': '房产',
        'stocks': '股票',
        'education': '教育',
        'science': '科技',
        'society': '社会',
        'politics': '时政',
        'sports': '体育',
        'game': '游戏',
        'entertainment': '娱乐'
    },

    # 选择使用的类别（论文中的5类）
    'selected_classes': ['entertainment', 'finance', 'education', 'sports', 'science'],

    # 采样配置
    'num_samples_per_category': 2000,  # 每类使用的样本数（None表示全部）
    'random_seed': 42
}
PREPROCESS_CONFIG = {
    'max_seq_length': 500,  # 最大文本长度
    'stopwords_path': 'stopwords.txt',  # 更新路径
    'custom_dict_path': None  # 自定义词典路径（可选）
}
TFIDF_CONFIG = {
    'max_features': 8000,
    'min_df': 2,
    'max_df': 0.8,
    'ngram_range': (1, 2)
}
WORD2VEC_CONFIG = {
    'vector_size': 100,
    'window': 5,
    'min_count': 5,
    'epochs': 10,
    'workers': 4,
    'sg': 1,  # Skip-gram
    'learning_rate': 0.025
}
BERT_CONFIG = {
    'model_name': 'bert-base-chinese',
    'max_length': 128,
    'batch_size': 32,
    'hidden_size': 768,
    'num_epochs': 3,
    'learning_rate': 2e-5
}
FUSION_CONFIG = {
    'tfidf_weight': 0.2,
    'word2vec_weight': 0.3,
    'bert_weight': 0.5,
    'standardization': 'zscore'  # 'zscore' or 'minmax'
}
BASE_MODELS_CONFIG = {
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    'svm': {
        'C': 10,
        'kernel': 'rbf',
        'gamma': 0.01,
        'class_weight': 'balanced'
    },
    'multinomial_nb': {
        'alpha': 1.0,
        'fit_prior': True
    },
    'knn': {
        'n_neighbors': 10,
        'metric': 'minkowski',
        'weights': 'distance'
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42
    },
    'gbdt': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'reg_lambda': 1,
        'subsample': 0.8,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'num_leaves': 63,
        'reg_lambda': 1,
        'random_state': 42
    }
}
ENSEMBLE_CONFIG = {
    'n_folds': 5,  
    'meta_model': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000
    }
}
TRAIN_CONFIG = {
    'save_model_path': 'models/',
    'save_feature_path': 'features/',
    'log_path': 'logs/',
    'random_seed': 42
}
EVAL_CONFIG = {
    'metrics': ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'micro_f1'],
    'confusion_matrix': True,
    'classification_report': True
}
def create_directories():
    dirs = [
        DATA_CONFIG['data_path'],
        TRAIN_CONFIG['save_model_path'],
        TRAIN_CONFIG['save_feature_path'],
        TRAIN_CONFIG['log_path']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

if __name__ == '__main__':
    create_directories()
    print("配置文件加载成功！")
    print(f"数据类别: {DATA_CONFIG['categories']}")
    print(f"基模型数量: {len(BASE_MODELS_CONFIG)}")
