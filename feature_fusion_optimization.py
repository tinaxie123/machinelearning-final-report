"""
特征融合与权重优化
实现TF-IDF、Word2Vec、BERT三种特征的加权融合
通过网格搜索和K折交叉验证选择最佳权重组合
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from itertools import product
from tqdm import tqdm
import json


def load_features():
    """加载三种特征并统一降维到100维"""
    print("\n" + "=" * 80)
    print("加载特征数据")
    print("=" * 80)

    # 加载TF-IDF特征并降维到100维
    print("\n[1/3] 加载TF-IDF特征...")
    with open('features/tfidf/tfidf_features.pkl', 'rb') as f:
        tfidf_data = pickle.load(f)
    X_train_tfidf = tfidf_data['X_train_tfidf']
    X_val_tfidf = tfidf_data['X_val_tfidf']
    X_test_tfidf = tfidf_data['X_test_tfidf']
    print(f"  原始维度 - 训练集: {X_train_tfidf.shape}")
    print(f"  原始维度 - 验证集: {X_val_tfidf.shape}")
    print(f"  原始维度 - 测试集: {X_test_tfidf.shape}")

    # 使用PCA降维到100维
    print("  使用PCA降维到100维...")
    pca_tfidf = PCA(n_components=100, random_state=42)
    # 检查是否为稀疏矩阵
    if hasattr(X_train_tfidf, 'toarray'):
        X_train_tfidf = pca_tfidf.fit_transform(X_train_tfidf.toarray())
        X_val_tfidf = pca_tfidf.transform(X_val_tfidf.toarray())
        X_test_tfidf = pca_tfidf.transform(X_test_tfidf.toarray())
    else:
        X_train_tfidf = pca_tfidf.fit_transform(X_train_tfidf)
        X_val_tfidf = pca_tfidf.transform(X_val_tfidf)
        X_test_tfidf = pca_tfidf.transform(X_test_tfidf)
    print(f"  降维后 - 训练集: {X_train_tfidf.shape}")
    print(f"  降维后 - 验证集: {X_val_tfidf.shape}")
    print(f"  降维后 - 测试集: {X_test_tfidf.shape}")
    print(f"  方差解释率: {pca_tfidf.explained_variance_ratio_.sum():.4f}")

    # 加载Word2Vec特征（已经是100维）
    print("\n[2/3] 加载Word2Vec特征...")
    with open('features/word2vec/word2vec_features.pkl', 'rb') as f:
        w2v_data = pickle.load(f)
    X_train_w2v = w2v_data['X_train_w2v']
    X_val_w2v = w2v_data['X_val_w2v']
    X_test_w2v = w2v_data['X_test_w2v']
    print(f"  训练集: {X_train_w2v.shape}")
    print(f"  验证集: {X_val_w2v.shape}")
    print(f"  测试集: {X_test_w2v.shape}")

    # 加载BERT特征（使用100维版本）
    print("\n[3/3] 加载BERT特征（100维版本）...")
    with open('features/bert/bert_features_pca_100d.pkl', 'rb') as f:
        bert_data = pickle.load(f)
    X_train_bert = bert_data['X_train_pca']
    X_val_bert = bert_data['X_val_pca']
    X_test_bert = bert_data['X_test_pca']
    print(f"  训练集: {X_train_bert.shape}")
    print(f"  验证集: {X_val_bert.shape}")
    print(f"  测试集: {X_test_bert.shape}")

    # 加载标签
    print("\n加载标签数据...")
    with open('features/preprocessed/preprocessed_data.pkl', 'rb') as f:
        label_data = pickle.load(f)
    y_train = label_data['y_train']
    y_val = label_data['y_val']
    y_test = label_data['y_test']
    print(f"  训练集标签: {len(y_train)}")
    print(f"  验证集标签: {len(y_val)}")
    print(f"  测试集标签: {len(y_test)}")

    return {
        'tfidf': (X_train_tfidf, X_val_tfidf, X_test_tfidf),
        'word2vec': (X_train_w2v, X_val_w2v, X_test_w2v),
        'bert': (X_train_bert, X_val_bert, X_test_bert),
        'labels': (y_train, y_val, y_test),
        'pca_tfidf': pca_tfidf
    }


def standardize_features(X_train, X_val, X_test):
    """
    Z-score标准化特征
    V' = (V - μ) / σ
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def fuse_features(features_list, weights):
    """
    加权融合特征（所有特征已统一降维到100维）
    V_fusion = w1 * V'_tfidf + w2 * V'_w2v + w3 * V'_bert

    Args:
        features_list: 特征列表 [(X_train1, X_val1, X_test1), ...]
        weights: 权重列表 [w1, w2, w3]

    Returns:
        融合后的特征 (X_train_fused, X_val_fused, X_test_fused)
    """
    # 初始化融合特征为零
    X_train_fused = None
    X_val_fused = None
    X_test_fused = None

    # 加权相加所有特征
    for i, (X_train, X_val, X_test) in enumerate(features_list):
        if X_train_fused is None:
            # 初始化
            X_train_fused = weights[i] * X_train
            X_val_fused = weights[i] * X_val
            X_test_fused = weights[i] * X_test
        else:
            # 加权相加
            X_train_fused += weights[i] * X_train
            X_val_fused += weights[i] * X_val
            X_test_fused += weights[i] * X_test

    return X_train_fused, X_val_fused, X_test_fused


def generate_weight_combinations(step=0.1):
    """
    生成所有满足约束的权重组合
    W = {(w1, w2, w3) | w1 + w2 + w3 = 1, wi ∈ {0, δ, 2δ, ..., 1}}

    Args:
        step: 权重步长，默认0.1

    Returns:
        权重组合列表
    """
    weights = []
    steps = int(1.0 / step) + 1

    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                w1 = i * step
                w2 = j * step
                w3 = k * step

                # 检查权重和是否为1（考虑浮点误差）
                if abs(w1 + w2 + w3 - 1.0) < 1e-6:
                    weights.append((round(w1, 1), round(w2, 1), round(w3, 1)))

    return weights


def evaluate_weight_combination(features_list, y_train, y_val, weights, model_type='lr'):
    """
    评估单个权重组合的性能

    Args:
        features_list: 标准化后的特征列表
        y_train: 训练集标签
        y_val: 验证集标签
        weights: 权重组合 (w1, w2, w3)
        model_type: 模型类型，默认'lr'（逻辑回归）

    Returns:
        验证集宏平均F1值
    """
    # 融合特征
    X_train_fused, X_val_fused, _ = fuse_features(features_list, weights)

    # 训练分类器
    if model_type == 'lr':
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    clf.fit(X_train_fused, y_train)

    # 预测并评估
    y_pred = clf.predict(X_val_fused)
    f1_macro = f1_score(y_val, y_pred, average='macro')

    return f1_macro


def grid_search_weights(features_list, y_train, y_val, step=0.1, model_type='lr'):
    """
    网格搜索最佳权重组合

    Args:
        features_list: 标准化后的特征列表
        y_train: 训练集标签
        y_val: 验证集标签
        step: 权重步长
        model_type: 模型类型

    Returns:
        最佳权重组合和对应的F1值
    """
    print("\n" + "=" * 80)
    print("网格搜索最佳权重组合")
    print("=" * 80)

    # 生成所有权重组合
    weight_combinations = generate_weight_combinations(step)
    print(f"\n总共有 {len(weight_combinations)} 个权重组合需要评估")
    print(f"权重约束: w1 + w2 + w3 = 1, wi ∈ {{0, {step}, {2*step}, ..., 1}}")

    # 评估每个权重组合
    best_weights = None
    best_f1 = 0.0
    results = []

    print("\n开始评估...")
    for weights in tqdm(weight_combinations, desc="评估权重组合"):
        f1 = evaluate_weight_combination(features_list, y_train, y_val, weights, model_type)
        results.append({
            'weights': weights,
            'f1_macro': f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights

    # 按F1值排序
    results.sort(key=lambda x: x['f1_macro'], reverse=True)

    print("\n" + "=" * 80)
    print("网格搜索完成！")
    print("=" * 80)
    print(f"\n最佳权重组合: TF-IDF={best_weights[0]}, Word2Vec={best_weights[1]}, BERT={best_weights[2]}")
    print(f"最佳验证集宏平均F1值: {best_f1:.4f}")

    print("\n前10个最佳权重组合:")
    print("-" * 60)
    print(f"{'排名':<6}{'TF-IDF':<10}{'Word2Vec':<12}{'BERT':<10}{'F1 (Macro)':<12}")
    print("-" * 60)
    for i, result in enumerate(results[:10], 1):
        w = result['weights']
        f1 = result['f1_macro']
        print(f"{i:<6}{w[0]:<10.1f}{w[1]:<12.1f}{w[2]:<10.1f}{f1:<12.4f}")

    return best_weights, best_f1, results


def k_fold_cross_validation(features_list, y_train, weights, k=5, model_type='lr'):
    """
    K折交叉验证评估权重组合

    Args:
        features_list: 标准化后的特征列表
        y_train: 训练集标签
        weights: 权重组合
        k: 折数
        model_type: 模型类型

    Returns:
        平均F1值和标准差
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []

    # 确保y_train是numpy数组
    y_train = np.array(y_train)

    # 首先融合所有训练数据
    X_train_fused, _, _ = fuse_features(features_list, weights)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_fused), 1):
        X_fold_train = X_train_fused[train_idx]
        X_fold_val = X_train_fused[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        # 训练分类器
        if model_type == 'lr':
            clf = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            )

        clf.fit(X_fold_train, y_fold_train)
        y_pred = clf.predict(X_fold_val)
        f1 = f1_score(y_fold_val, y_pred, average='macro')
        f1_scores.append(f1)

    return np.mean(f1_scores), np.std(f1_scores)


def save_results(best_weights, best_f1, all_results, output_dir='features/fusion'):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存最佳权重
    result_dict = {
        'best_weights': {
            'tfidf': best_weights[0],
            'word2vec': best_weights[1],
            'bert': best_weights[2]
        },
        'best_f1_macro': float(best_f1),
        'all_results': [
            {
                'weights': {
                    'tfidf': r['weights'][0],
                    'word2vec': r['weights'][1],
                    'bert': r['weights'][2]
                },
                'f1_macro': float(r['f1_macro'])
            }
            for r in all_results
        ]
    }

    output_path = os.path.join(output_dir, 'weight_optimization_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存至: {output_path}")


def main():
    print("\n" + "=" * 80)
    print("特征融合与权重优化实验")
    print("=" * 80)
    print("\n实验方法:")
    print("  1. 统一降维: 将所有特征降维到100维（TF-IDF和BERT使用PCA）")
    print("  2. Z-score标准化各特征（消除量纲差异）")
    print("  3. 加权融合: V_fusion = w1*V'_tfidf + w2*V'_w2v + w3*V'_bert")
    print("  4. 网格搜索最佳权重（步长0.1）")
    print("  5. 目标: 最大化验证集宏平均F1值")

    # 1. 加载特征并降维
    data = load_features()
    pca_tfidf = data['pca_tfidf']

    # 2. Z-score标准化
    print("\n" + "=" * 80)
    print("特征标准化（Z-score）")
    print("=" * 80)

    print("\n[1/3] 标准化TF-IDF特征...")
    X_train_tfidf_std, X_val_tfidf_std, X_test_tfidf_std, scaler_tfidf = \
        standardize_features(*data['tfidf'])
    print(f"  均值: {X_train_tfidf_std.mean():.6f}, 标准差: {X_train_tfidf_std.std():.6f}")

    print("\n[2/3] 标准化Word2Vec特征...")
    X_train_w2v_std, X_val_w2v_std, X_test_w2v_std, scaler_w2v = \
        standardize_features(*data['word2vec'])
    print(f"  均值: {X_train_w2v_std.mean():.6f}, 标准差: {X_train_w2v_std.std():.6f}")

    print("\n[3/3] 标准化BERT特征...")
    X_train_bert_std, X_val_bert_std, X_test_bert_std, scaler_bert = \
        standardize_features(*data['bert'])
    print(f"  均值: {X_train_bert_std.mean():.6f}, 标准差: {X_train_bert_std.std():.6f}")

    # 组织标准化后的特征
    features_list = [
        (X_train_tfidf_std, X_val_tfidf_std, X_test_tfidf_std),
        (X_train_w2v_std, X_val_w2v_std, X_test_w2v_std),
        (X_train_bert_std, X_val_bert_std, X_test_bert_std)
    ]

    y_train, y_val, y_test = data['labels']

    # 3. 网格搜索最佳权重
    best_weights, best_f1, all_results = grid_search_weights(
        features_list, y_train, y_val, step=0.1, model_type='lr'
    )

    # 4. K折交叉验证（可选，评估最佳权重的稳定性）
    print("\n" + "=" * 80)
    print("K折交叉验证（评估最佳权重稳定性）")
    print("=" * 80)

    mean_f1, std_f1 = k_fold_cross_validation(
        features_list, y_train, best_weights, k=5, model_type='lr'
    )
    print(f"\n5折交叉验证结果:")
    print(f"  平均宏平均F1值: {mean_f1:.4f} ± {std_f1:.4f}")

    # 5. 保存结果
    save_results(best_weights, best_f1, all_results)

    # 6. 保存标准化器、PCA模型和最佳权重（用于后续训练）
    fusion_dir = 'features/fusion'
    os.makedirs(fusion_dir, exist_ok=True)

    with open(os.path.join(fusion_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump({
            'tfidf': scaler_tfidf,
            'word2vec': scaler_w2v,
            'bert': scaler_bert
        }, f)

    with open(os.path.join(fusion_dir, 'pca_models.pkl'), 'wb') as f:
        pickle.dump({
            'tfidf': pca_tfidf
        }, f)

    print("\n标准化器已保存至: features/fusion/scalers.pkl")
    print("PCA模型已保存至: features/fusion/pca_models.pkl")

    # 7. 使用最佳权重生成融合特征并保存
    print("\n" + "=" * 80)
    print("生成并保存最佳融合特征")
    print("=" * 80)

    X_train_fused, X_val_fused, X_test_fused = fuse_features(features_list, best_weights)

    fused_features = {
        'X_train': X_train_fused,
        'X_val': X_val_fused,
        'X_test': X_test_fused,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'weights': best_weights
    }

    fusion_path = os.path.join(fusion_dir, 'fused_features.pkl')
    with open(fusion_path, 'wb') as f:
        pickle.dump(fused_features, f)

    print(f"\n融合特征已保存至: {fusion_path}")
    print(f"  训练集: {X_train_fused.shape}")
    print(f"  验证集: {X_val_fused.shape}")
    print(f"  测试集: {X_test_fused.shape}")

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"\n最终结果:")
    print(f"  最佳权重: TF-IDF={best_weights[0]}, Word2Vec={best_weights[1]}, BERT={best_weights[2]}")
    print(f"  验证集宏平均F1值: {best_f1:.4f}")
    print(f"  交叉验证F1值: {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == '__main__':
    main()
