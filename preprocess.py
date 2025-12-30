"""
步骤1：数据预处理和分词
按照论文方法进行文本清洗、jieba分词、停用词过滤
"""

import os
import sys
import pickle
import jieba
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_thucnews import THUCNewsLoader
from text_preprocessing import TextPreprocessor
from config import PREPROCESS_CONFIG, DATA_CONFIG, TRAIN_CONFIG

def main():
    print("\n" + "=" * 80)
    print("步骤1：数据预处理和分词")
    print("=" * 80)

    # 创建保存目录
    save_dir = os.path.join(TRAIN_CONFIG['save_feature_path'], 'preprocessed')
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载数据
    print("\n[1/3] 加载THUCNews数据集...")
    loader = THUCNewsLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_train_dev_test()

    # 2. 初始化预处理器
    print("\n[2/3] 初始化文本预处理器...")
    print(f"  停用词路径: {PREPROCESS_CONFIG['stopwords_path']}")
    print(f"  自定义词典: {PREPROCESS_CONFIG.get('custom_dict_path', '无')}")

    preprocessor = TextPreprocessor(
        custom_dict_path=PREPROCESS_CONFIG.get('custom_dict_path'),
        stopwords_path=PREPROCESS_CONFIG['stopwords_path']
    )

    # 3. 文本预处理（清洗+分词）
    print("\n[3/3] 文本预处理和分词...")
    print("=" * 60)

    print("\n处理训练集...")
    X_train_tokens = []
    for text in tqdm(X_train, desc="  训练集"):
        tokens = preprocessor.preprocess(text, use_stopwords=True)
        X_train_tokens.append(tokens)

    print("\n处理验证集...")
    X_val_tokens = []
    for text in tqdm(X_val, desc="  验证集"):
        tokens = preprocessor.preprocess(text, use_stopwords=True)
        X_val_tokens.append(tokens)

    print("\n处理测试集...")
    X_test_tokens = []
    for text in tqdm(X_test, desc="  测试集"):
        tokens = preprocessor.preprocess(text, use_stopwords=True)
        X_test_tokens.append(tokens)

    # 4. 保存分词结果
    print("\n" + "=" * 60)
    print("保存预处理结果...")

    preprocessed_data = {
        'X_train_tokens': X_train_tokens,
        'X_val_tokens': X_val_tokens,
        'X_test_tokens': X_test_tokens,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    save_path = os.path.join(save_dir, 'preprocessed_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    print(f"[OK] 保存至: {save_path}")

    # 5. 输出统计信息
    print("\n" + "=" * 80)
    print("预处理完成！统计信息：")
    print("=" * 80)

    print(f"\n训练集:")
    print(f"  样本数: {len(X_train_tokens)}")
    print(f"  平均词数: {sum(len(tokens) for tokens in X_train_tokens) / len(X_train_tokens):.2f}")

    print(f"\n验证集:")
    print(f"  样本数: {len(X_val_tokens)}")
    print(f"  平均词数: {sum(len(tokens) for tokens in X_val_tokens) / len(X_val_tokens):.2f}")

    print(f"\n测试集:")
    print(f"  样本数: {len(X_test_tokens)}")
    print(f"  平均词数: {sum(len(tokens) for tokens in X_test_tokens) / len(X_test_tokens):.2f}")

    # 6. 显示分词示例
    print("\n" + "=" * 80)
    print("分词示例（前3条训练样本）：")
    print("=" * 80)

    for i in range(min(3, len(X_train_tokens))):
        print(f"\n样本 {i+1}:")
        print(f"  类别: {y_train[i]}")
        print(f"  原文: {X_train[i][:80]}...")
        print(f"  分词: {' / '.join(X_train_tokens[i][:20])}{'...' if len(X_train_tokens[i]) > 20 else ''}")
        print(f"  词数: {len(X_train_tokens[i])}")

    print("\n" + "=" * 80)
    print("[完成] 步骤1完成！可以继续执行步骤2（TF-IDF特征提取）")
    print("=" * 80)

    return preprocessed_data


if __name__ == '__main__':
    main()
