# feature_engineering.py - Create new features from existing data
import pandas as pd

def add_datetime_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    return df

def add_ratio_feature(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    df[f'{col_a}_to_{col_b}_ratio'] = df[col_a] / (df[col_b] + 1e-9)
    return df
