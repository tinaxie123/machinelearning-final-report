"""
主程序：完整的训练、评估和预测流程
"""

import os
import argparse
import numpy as np
import pickle
from typing import List

from config import (
    DATA_CONFIG, PREPROCESS_CONFIG, TFIDF_CONFIG,
    WORD2VEC_CONFIG, BERT_CONFIG, FUSION_CONFIG,
    BASE_MODELS_CONFIG, ENSEMBLE_CONFIG, TRAIN_CONFIG,
    create_directories
)
from text_preprocessing import TextPreprocessor
from load_thucnews import THUCNewsLoader
from feature_extraction import TFIDFExtractor, Word2VecExtractor, BERTExtractor
from feature_fusion import FeatureFusion, FeatureManager
from base_models import BaseModelManager
from ensemble import EnsembleManager
from evaluate import ModelEvaluator, ComparisonEvaluator


class ChineseNewsClassifier:
    """中文新闻文本分类系统"""

    def __init__(self):
        """初始化分类系统"""
        # 创建必要目录
        create_directories()

        # 初始化各模块
        self.preprocessor = TextPreprocessor(
            custom_dict_path=PREPROCESS_CONFIG['custom_dict_path'],
            stopwords_path=PREPROCESS_CONFIG['stopwords_path']
        )

        self.tfidf_extractor = TFIDFExtractor(**TFIDF_CONFIG)
        self.word2vec_extractor = Word2VecExtractor(**WORD2VEC_CONFIG)
        self.bert_extractor = BERTExtractor(**BERT_CONFIG)

        self.fusion = FeatureFusion(**FUSION_CONFIG)

        self.feature_manager = FeatureManager(
            self.tfidf_extractor,
            self.word2vec_extractor,
            self.bert_extractor,
            self.fusion
        )

        self.base_model_manager = BaseModelManager(BASE_MODELS_CONFIG)
        self.ensemble_manager = EnsembleManager(
            self.base_model_manager,
            n_folds=ENSEMBLE_CONFIG['n_folds']
        )

        # 数据
        self.X_train_tokens = None
        self.X_val_tokens = None
        self.X_test_tokens = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.X_train_fused = None
        self.X_val_fused = None
        self.X_test_fused = None

    def load_and_preprocess_data(self):
        """加载和预处理THUCNews数据"""
        print("\n" + "=" * 80)
        print("步骤 1: 加载和预处理数据")
        print("=" * 80)

        # 使用THUCNewsLoader加载数据
        loader = THUCNewsLoader()
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_train_dev_test()

        # 文本预处理（分词）
        print("\n文本预处理（分词）...")
        print("=" * 60)

        print("\n处理训练集...")
        self.X_train_tokens = self.preprocessor.preprocess_batch(X_train)

        print("\n处理验证集...")
        self.X_val_tokens = self.preprocessor.preprocess_batch(X_val)

        print("\n处理测试集...")
        self.X_test_tokens = self.preprocessor.preprocess_batch(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"\n预处理完成！")
        print(f"训练集: {len(self.X_train_tokens)} 样本")
        print(f"验证集: {len(self.X_val_tokens)} 样本")
        print(f"测试集: {len(self.X_test_tokens)} 样本")
        print("=" * 60)

    def extract_and_fuse_features(self):
        """提取和融合特征"""
        print("\n" + "=" * 80)
        print("步骤 2: 特征提取与融合")
        print("=" * 80)

        # 训练集特征提取与融合
        print("\n处理训练集...")
        train_features, self.X_train_fused = self.feature_manager.extract_and_fuse(
            self.X_train_tokens, fit=True
        )

        # 验证集特征提取与融合
        print("\n处理验证集...")
        val_features, self.X_val_fused = self.feature_manager.extract_and_fuse(
            self.X_val_tokens, fit=False
        )

        # 测试集特征提取与融合
        print("\n处理测试集...")
        test_features, self.X_test_fused = self.feature_manager.extract_and_fuse(
            self.X_test_tokens, fit=False
        )

        print("\n特征提取与融合完成！")

    def train_ensemble_models(self):
        """训练集成模型"""
        print("\n" + "=" * 80)
        print("步骤 3: 训练集成模型")
        print("=" * 80)

        self.ensemble_manager.train(
            self.X_train_fused, self.y_train,
            self.X_val_fused, self.y_val
        )

        print("\n集成模型训练完成！")

    def evaluate_models(self):
        """评估模型"""
        print("\n" + "=" * 80)
        print("步骤 4: 模型评估")
        print("=" * 80)

        comparison = ComparisonEvaluator()

        # 评估各基模型
        print("\n评估基模型...")
        for model_name in self.base_model_manager.get_model_names():
            y_pred = self.base_model_manager.predict_single_model(
                model_name, self.X_test_fused
            )
            comparison.add_result(model_name, self.y_test, y_pred)

        # 评估加权软投票
        print("\n评估加权软投票...")
        y_pred_voting = self.ensemble_manager.predict_weighted_voting(self.X_test_fused)
        comparison.add_result("Weighted_Soft_Voting", self.y_test, y_pred_voting)

        evaluator_voting = ModelEvaluator(self.y_test, y_pred_voting, "加权软投票")
        evaluator_voting.print_metrics()
        evaluator_voting.print_classification_report()

        # 评估 Stacking
        print("\n评估 Stacking...")
        y_pred_stacking = self.ensemble_manager.predict_stacking(self.X_test_fused)
        comparison.add_result("Stacking", self.y_test, y_pred_stacking)

        evaluator_stacking = ModelEvaluator(self.y_test, y_pred_stacking, "Stacking")
        evaluator_stacking.print_metrics()
        evaluator_stacking.print_classification_report()

        # 打印对比表格
        comparison.print_comparison_table()

        # 找出最优模型
        best_model = comparison.get_best_model()
        print(f"\n最优模型: {best_model}")

        return comparison

    def save_models(self):
        """保存所有模型"""
        print("\n" + "=" * 80)
        print("保存模型...")
        print("=" * 80)

        # 保存特征提取器
        self.tfidf_extractor.save(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'tfidf_extractor.pkl')
        )
        self.word2vec_extractor.save(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'word2vec_extractor.model')
        )
        self.bert_extractor.save(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'bert_extractor')
        )

        # 保存特征融合器
        self.fusion.save(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'fusion.pkl')
        )

        # 保存基模型
        self.base_model_manager.save_models(
            os.path.join(TRAIN_CONFIG['save_model_path'], 'base_models.pkl')
        )

        # 保存集成模型
        self.ensemble_manager.save(
            os.path.join(TRAIN_CONFIG['save_model_path'], 'ensemble.pkl')
        )

        print("\n所有模型保存完成！")

    def load_models(self):
        """加载所有模型"""
        print("\n" + "=" * 80)
        print("加载模型...")
        print("=" * 80)

        # 加载特征提取器
        self.tfidf_extractor.load(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'tfidf_extractor.pkl')
        )
        self.word2vec_extractor.load(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'word2vec_extractor.model')
        )
        self.bert_extractor.load(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'bert_extractor')
        )

        # 加载特征融合器
        self.fusion.load(
            os.path.join(TRAIN_CONFIG['save_feature_path'], 'fusion.pkl')
        )

        # 加载基模型
        self.base_model_manager.load_models(
            os.path.join(TRAIN_CONFIG['save_model_path'], 'base_models.pkl')
        )

        # 加载集成模型
        self.ensemble_manager.load(
            os.path.join(TRAIN_CONFIG['save_model_path'], 'ensemble.pkl')
        )

        print("\n所有模型加载完成！")

    def predict(self, text: str, method='stacking') -> str:
        """
        预测单条文本

        Args:
            text: 输入文本
            method: 预测方法 ('stacking' 或 'voting')

        Returns:
            预测类别
        """
        # 预处理
        tokens = self.preprocessor.preprocess(text)

        # 特征提取与融合
        _, fused_features = self.feature_manager.extract_and_fuse([tokens], fit=False)

        # 预测
        if method == 'stacking':
            y_pred = self.ensemble_manager.predict_stacking(fused_features)
        elif method == 'voting':
            y_pred = self.ensemble_manager.predict_weighted_voting(fused_features)
        else:
            raise ValueError(f"不支持的预测方法: {method}")

        return y_pred[0]

    def run_full_pipeline(self):
        """运行完整流程"""
        self.load_and_preprocess_data()
        self.extract_and_fuse_features()
        self.train_ensemble_models()
        comparison = self.evaluate_models()
        self.save_models()

        return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中文新闻文本分类系统')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='运行模式')
    parser.add_argument('--text', type=str, default=None,
                        help='预测文本（仅在 predict 模式下使用）')
    parser.add_argument('--method', type=str, default='stacking',
                        choices=['stacking', 'voting'],
                        help='预测方法（仅在 predict 模式下使用）')

    args = parser.parse_args()

    # 初始化分类器
    classifier = ChineseNewsClassifier()

    if args.mode == 'train':
        print("\n" + "=" * 80)
        print("训练模式")
        print("=" * 80)
        classifier.run_full_pipeline()

    elif args.mode == 'evaluate':
        print("\n" + "=" * 80)
        print("评估模式")
        print("=" * 80)
        classifier.load_models()
        classifier.load_and_preprocess_data()
        classifier.extract_and_fuse_features()
        classifier.evaluate_models()

    elif args.mode == 'predict':
        print("\n" + "=" * 80)
        print("预测模式")
        print("=" * 80)

        if args.text is None:
            print("错误: 请提供预测文本（--text）")
            return

        classifier.load_models()
        prediction = classifier.predict(args.text, method=args.method)

        print(f"\n输入文本: {args.text}")
        print(f"预测类别: {prediction}")
        print(f"预测方法: {args.method}")


if __name__ == '__main__':
    main()
