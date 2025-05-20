# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import joblib # 用于保存模型
from typing import Tuple # 用于类型注解

from data.preprocessing import load_data, encode_categorical_features, handle_missing_values

def prepare_data() -> Tuple:
    """准备训练数据

    Returns:
        训练集和测试集的特征和标签
    """
    # 加载和预处理数据
    data = load_data("data/raw/data.csv")
    data_encoded, _ = encode_categorical_features(data)
    data_clean = handle_missing_values(data_encoded)
    
    # 分离特征和标签
    X = data_clean.drop(['Credit Default'], axis=1)
    y = data_clean['Credit Default']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_params=None) -> RandomForestClassifier:
    """训练随机森林模型

    Args:
        X_train: 训练特征
        y_train: 训练标签
        model_params: 模型参数字典

    Returns:
        训练好的模型
    """
    if model_params is None:
        model_params = {'random_state': 42}
    
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test) -> None:
    """评估模型性能

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
    """
    y_pred = model.predict(X_test)
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, model_path: str) -> None:
    """保存模型

    Args:
        model: 训练好的模型
        model_path: 模型保存路径
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n模型已保存至: {model_path}")

if __name__ == "__main__":
    # 准备数据
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 记录结束时间
    end_time = time.time()
    print(f"\n训练耗时: {end_time - start_time:.4f} 秒")
    
    # 评估模型
    evaluate_model(model, X_test, y_test)
    
    # 保存模型
    save_model(model, "models/random_forest_model.joblib") 