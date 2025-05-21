import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对数据进行基础清洗：
    - 去除完全空的行和列
    - 填充简单的缺失值
    - 去除重复行
    """

    # 去除完全空的行和列
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    
    # 简单方式：缺失值可以使用列均值或中位数或固定值进行填充
    # 根据情况灵活选择，这里仅演示：
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # 如果有字符串列缺失值，可以考虑填充空字符串或 'Unknown'
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)

    # 去除重复行
    df.drop_duplicates(inplace=True)

    return df
