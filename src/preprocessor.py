# preprocessor.py - Clean, encode, and scale data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
