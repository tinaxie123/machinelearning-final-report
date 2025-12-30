"""
THUCNews数据集加载器 - 处理单文件格式
"""

import os
import numpy as np
from typing import List, Tuple
from collections import Counter
from tqdm import tqdm
from config import DATA_CONFIG


class THUCNewsLoader:
    """THUCNews数据集加载器"""

    def __init__(self):
        """初始化加载器"""
        self.data_path = DATA_CONFIG['data_path']
        self.class_mapping = DATA_CONFIG['class_mapping']
        self.class_mapping_cn = DATA_CONFIG['class_mapping_cn']
        self.selected_classes = DATA_CONFIG['selected_classes']
        self.num_samples_per_category = DATA_CONFIG['num_samples_per_category']
        self.random_seed = DATA_CONFIG['random_seed']

    def load_file(self, filename: str) -> Tuple[List[str], List[str]]:
        """
        加载单个数据文件

        Args:
            filename: 文件名（train.txt, dev.txt, test.txt）

        Returns:
            (文本列表, 标签列表)
        """
        filepath = os.path.join(self.data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        print(f"\n加载文件: {filepath}")

        texts = []
        labels = []
        category_counts = Counter()

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  读取中"):
                line = line.strip()
                if not line:
                    continue

                # 格式：文本\t类别编号
                parts = line.split('\t')
                if len(parts) != 2:
                    continue

                text, label_id = parts

                # 转换标签编号为类别名
                try:
                    label_id = int(label_id)
                    if label_id not in self.class_mapping:
                        continue

                    class_name = self.class_mapping[label_id]

                    # 只选择指定的类别
                    if class_name in self.selected_classes:
                        texts.append(text)
                        # 使用中文标签
                        cn_label = self.class_mapping_cn[class_name]
                        labels.append(cn_label)
                        category_counts[cn_label] += 1

                except ValueError:
                    continue

        print(f"\n[OK] 加载完成，总计 {len(texts)} 条样本")
        print("\n类别分布:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count} 条")

        return texts, labels

    def load_and_sample(self, filename: str) -> Tuple[List[str], List[str]]:
        """
        加载数据并进行采样

        Args:
            filename: 文件名

        Returns:
            (文本列表, 标签列表)
        """
        # 先加载全部数据
        all_texts, all_labels = self.load_file(filename)

        # 如果不需要采样，直接返回
        if self.num_samples_per_category is None:
            return all_texts, all_labels

        # 按类别分组
        category_data = {}
        for text, label in zip(all_texts, all_labels):
            if label not in category_data:
                category_data[label] = []
            category_data[label].append(text)

        # 每个类别采样
        np.random.seed(self.random_seed)
        sampled_texts = []
        sampled_labels = []

        print(f"\n采样配置: 每类 {self.num_samples_per_category} 条")
        print("=" * 60)

        for category in sorted(category_data.keys()):
            texts_in_category = category_data[category]

            # 随机采样
            if len(texts_in_category) > self.num_samples_per_category:
                indices = np.random.choice(
                    len(texts_in_category),
                    self.num_samples_per_category,
                    replace=False
                )
                selected_texts = [texts_in_category[i] for i in indices]
            else:
                selected_texts = texts_in_category

            sampled_texts.extend(selected_texts)
            sampled_labels.extend([category] * len(selected_texts))

            print(f"{category}: {len(selected_texts)} 条")

        print("=" * 60)
        print(f"总计: {len(sampled_texts)} 条样本\n")

        return sampled_texts, sampled_labels

    def load_train_dev_test(self) -> Tuple:
        """
        加载训练集、验证集、测试集

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n" + "=" * 80)
        print("加载 THUCNews 数据集")
        print("=" * 80)
        print(f"数据路径: {self.data_path}")
        print(f"使用类别: {[self.class_mapping_cn[c] for c in self.selected_classes]}")
        print("=" * 80)

        # 加载训练集
        print("\n[1/3] 加载训练集...")
        X_train, y_train = self.load_and_sample(DATA_CONFIG['train_file'])

        # 加载验证集
        print("\n[2/3] 加载验证集...")
        X_val, y_val = self.load_and_sample(DATA_CONFIG['dev_file'])

        # 加载测试集
        print("\n[3/3] 加载测试集...")
        X_test, y_test = self.load_and_sample(DATA_CONFIG['test_file'])

        print("\n" + "=" * 80)
        print("数据集加载完成！")
        print("=" * 80)
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        print("=" * 80)

        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    # 测试数据加载
    loader = THUCNewsLoader()

    # 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_train_dev_test()

    # 显示示例
    print("\n训练集示例:")
    for i in range(min(3, len(X_train))):
        print(f"\n样本 {i+1}:")
        print(f"类别: {y_train[i]}")
        print(f"文本: {X_train[i][:100]}...")
