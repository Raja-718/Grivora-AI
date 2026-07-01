# test_model.py
import pytest
import pandas as pd
from src.preprocessor import clean_data, encode_categoricals

def test_clean_data():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0

def test_encode_categoricals():
    df = pd.DataFrame({"cat": ["a", "b", "a"]})
    encoded = encode_categoricals(df)
    assert encoded["cat"].dtype in ["int32", "int64"]
