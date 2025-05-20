import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from typing import Any

def plot_feature_importance_shap(model: Any, X_test, save_path: str = None) -> None:
    """绘制SHAP特征重要性图

    Args:
        model: 训练好的模型
        X_test: 测试数据
        save_path: 图片保存路径
    """
    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 绘制特征重要性条形图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[:, :, 0], X_test, plot_type="bar", show=False)
    plt.title("SHAP特征重要性")
    
    if save_path:
        plt.savefig(save_path)
        print(f"特征重要性图已保存至: {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path: str = None) -> None:
    """绘制混淆矩阵热力图

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 图片保存路径
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵图已保存至: {save_path}")
    plt.show()

def set_plot_style():
    """设置绘图样式"""
    plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 设置绘图样式
    set_plot_style()
    
    # 这里可以添加测试代码
    print("可视化模块加载成功！") 