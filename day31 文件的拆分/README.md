# 信用违约预测模型

这个项目实现了一个信用违约预测模型，使用随机森林算法对客户是否会发生信用违约进行预测。

## 项目结构

```
credit_default_prediction/
│
├── data/                   # 数据文件夹
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
│
├── src/                   # 源代码
│   ├── __init__.py
│   ├── data/             # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/           # 模型相关代码
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── visualization/    # 可视化相关代码
│       ├── __init__.py
│       └── plots.py
│
├── notebooks/            # Jupyter notebooks
│   └── model_development.ipynb
│
├── requirements.txt      # 项目依赖
└── README.md            # 项目说明文档
```

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行模型训练：

```bash
python src/models/train.py
```

## 详细说明

### 数据预处理

- 处理缺失值
- 特征编码（标签编码和独热编码）
- 数据集划分

### 模型训练

- 使用随机森林分类器
- 包含默认参数训练
- SHAP值解释模型预测

### 特征工程

- 连续特征处理
- 离散特征编码
- 特征重要性分析

## 注意事项

1. 所有模块导入都使用相对导入或绝对导入
2. 主要执行文件都包含 `if __name__ == "__main__":` 语句
3. 配置文件分离，避免硬编码
4. 使用日志记录而不是print语句

## 依赖说明

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
