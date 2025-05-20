import pandas as pd
import numpy as np
from typing import Tuple, Dict

def load_data(file_path: str) -> pd.DataFrame:
    """加载数据文件

    Args:
        file_path: 数据文件路径

    Returns:
        加载的数据框
    """
    return pd.read_csv(file_path)

def encode_categorical_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """对分类特征进行编码

    Args:
        data: 原始数据框

    Returns:
        编码后的数据框和编码映射字典
    """
    # Home Ownership 标签编码
    home_ownership_mapping = {
        'Own Home': 1,
        'Rent': 2,
        'Have Mortgage': 3,
        'Home Mortgage': 4
    }
    
    # Years in current job 标签编码
    years_in_job_mapping = {
        '< 1 year': 1,
        '1 year': 2,
        '2 years': 3,
        '3 years': 4,
        '4 years': 5,
        '5 years': 6,
        '6 years': 7,
        '7 years': 8,
        '8 years': 9,
        '9 years': 10,
        '10+ years': 11
    }
    
    # Term 映射
    term_mapping = {
        'Short Term': 0,
        'Long Term': 1
    }
    
    # 应用映射
    data_encoded = data.copy()
    data_encoded['Home Ownership'] = data['Home Ownership'].map(home_ownership_mapping)
    data_encoded['Years in current job'] = data['Years in current job'].map(years_in_job_mapping)
    data_encoded['Term'] = data['Term'].map(term_mapping)
    data_encoded.rename(columns={'Term': 'Long Term'}, inplace=True)
    
    # Purpose 独热编码
    data_encoded = pd.get_dummies(data_encoded, columns=['Purpose'])
    
    # 将独热编码列转换为整数类型
    purpose_columns = [col for col in data_encoded.columns if col not in data.columns]
    for col in purpose_columns:
        data_encoded[col] = data_encoded[col].astype(int)
    
    mappings = {
        'home_ownership': home_ownership_mapping,
        'years_in_job': years_in_job_mapping,
        'term': term_mapping
    }
    
    return data_encoded, mappings

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值

    Args:
        data: 包含缺失值的数据框

    Returns:
        处理后的数据框
    """
    data_clean = data.copy()
    continuous_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for feature in continuous_features:
        mode_value = data[feature].mode()[0]
        data_clean[feature].fillna(mode_value, inplace=True)
    
    return data_clean

if __name__ == "__main__":
    # 测试代码
    data = load_data("data/raw/data.csv")
    data_encoded, mappings = encode_categorical_features(data)
    data_clean = handle_missing_values(data_encoded)
    print("数据预处理完成！") 