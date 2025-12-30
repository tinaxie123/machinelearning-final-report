"""
评估模块：模型性能评估与可视化
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, y_true: List[str], y_pred: List[str], model_name: str = "Model"):
        """
        初始化评估器

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name

    def calculate_metrics(self) -> Dict[str, float]:
        """
        计算所有评估指标

        Returns:
            评估指标字典
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'macro_precision': precision_score(self.y_true, self.y_pred, average='macro'),
            'macro_recall': recall_score(self.y_true, self.y_pred, average='macro'),
            'macro_f1': f1_score(self.y_true, self.y_pred, average='macro'),
            'micro_f1': f1_score(self.y_true, self.y_pred, average='micro'),
            'weighted_f1': f1_score(self.y_true, self.y_pred, average='weighted')
        }
        return metrics

    def print_metrics(self):
        """打印评估指标"""
        metrics = self.calculate_metrics()

        print("\n" + "=" * 60)
        print(f"{self.model_name} 性能评估结果")
        print("=" * 60)
        print(f"准确率 (Accuracy):          {metrics['accuracy']:.4f}")
        print(f"宏平均精确率 (Macro-P):     {metrics['macro_precision']:.4f}")
        print(f"宏平均召回率 (Macro-R):     {metrics['macro_recall']:.4f}")
        print(f"宏平均 F1 (Macro-F1):      {metrics['macro_f1']:.4f}")
        print(f"微平均 F1 (Micro-F1):      {metrics['micro_f1']:.4f}")
        print(f"加权 F1 (Weighted-F1):     {metrics['weighted_f1']:.4f}")
        print("=" * 60)

    def print_classification_report(self):
        """打印分类报告"""
        print("\n" + "=" * 60)
        print(f"{self.model_name} 分类报告")
        print("=" * 60)
        print(classification_report(self.y_true, self.y_pred, digits=4))

    def plot_confusion_matrix(self, save_path: str = None, figsize=(10, 8)):
        """
        绘制混淆矩阵

        Args:
            save_path: 保存路径
            figsize: 图片大小
        """
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_true, self.y_pred)

        # 获取类别标签
        labels = sorted(list(set(self.y_true)))

        # 绘制热图
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")

        plt.show()

    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        获取每个类别的评估指标

        Returns:
            每个类别的指标字典
        """
        labels = sorted(list(set(self.y_true)))

        per_class_metrics = {}

        for label in labels:
            # 将多分类转换为二分类
            y_true_binary = [1 if y == label else 0 for y in self.y_true]
            y_pred_binary = [1 if y == label else 0 for y in self.y_pred]

            per_class_metrics[label] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
            }

        return per_class_metrics

    def print_per_class_metrics(self):
        """打印每个类别的评估指标"""
        per_class_metrics = self.get_per_class_metrics()

        print("\n" + "=" * 60)
        print(f"{self.model_name} 各类别性能")
        print("=" * 60)
        print(f"{'类别':<10} {'精确率':<12} {'召回率':<12} {'F1值':<12}")
        print("-" * 60)

        for label, metrics in per_class_metrics.items():
            print(f"{label:<10} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

        print("=" * 60)


class ComparisonEvaluator:
    """模型对比评估器"""

    def __init__(self):
        """初始化对比评估器"""
        self.results = {}

    def add_result(self, model_name: str, y_true: List[str], y_pred: List[str]):
        """
        添加模型结果

        Args:
            model_name: 模型名称
            y_true: 真实标签
            y_pred: 预测标签
        """
        evaluator = ModelEvaluator(y_true, y_pred, model_name)
        metrics = evaluator.calculate_metrics()
        self.results[model_name] = metrics

    def print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            print("暂无结果可对比")
            return

        print("\n" + "=" * 100)
        print("模型性能对比")
        print("=" * 100)
        print(f"{'模型名称':<20} {'准确率':<12} {'宏平均P':<12} "
              f"{'宏平均R':<12} {'宏平均F1':<12} {'微平均F1':<12}")
        print("-" * 100)

        for model_name, metrics in self.results.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['macro_precision']:<12.4f} "
                  f"{metrics['macro_recall']:<12.4f} "
                  f"{metrics['macro_f1']:<12.4f} "
                  f"{metrics['micro_f1']:<12.4f}")

        print("=" * 100)

    def plot_comparison_bar(self, metric='macro_f1', save_path: str = None, figsize=(12, 6)):
        """
        绘制模型对比柱状图

        Args:
            metric: 对比指标
            save_path: 保存路径
            figsize: 图片大小
        """
        if not self.results:
            print("暂无结果可对比")
            return

        # 提取数据
        model_names = list(self.results.keys())
        metric_values = [self.results[name][metric] for name in model_names]

        # 绘制柱状图
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(model_names)), metric_values, color='skyblue', edgecolor='navy')

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{value:.4f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存至: {save_path}")

        plt.show()

    def get_best_model(self, metric='macro_f1') -> str:
        """
        获取最优模型

        Args:
            metric: 评估指标

        Returns:
            最优模型名称
        """
        if not self.results:
            return None

        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0]


if __name__ == '__main__':
    # 测试评估功能
    y_true = ['娱乐', '财经', '教育', '体育', '科技'] * 20
    y_pred = ['娱乐', '财经', '教育', '体育', '娱乐'] * 20

    # 单模型评估
    evaluator = ModelEvaluator(y_true, y_pred, "测试模型")
    evaluator.print_metrics()
    evaluator.print_classification_report()
    evaluator.print_per_class_metrics()

    # 模型对比
    comparison = ComparisonEvaluator()
    comparison.add_result("模型A", y_true, y_pred)
    comparison.add_result("模型B", y_true, ['娱乐'] * 100)
    comparison.print_comparison_table()

    print(f"\n最优模型: {comparison.get_best_model()}")
