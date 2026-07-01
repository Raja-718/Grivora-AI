# test_data_loader.py
import pytest
import pandas as pd
from src.data_loader import load_file, get_summary

def test_load_csv(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("a,b\n1,2\n3,4")
    df = load_file(str(f))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]

def test_get_summary():
    df = pd.DataFrame({"x": [1, 2, None], "y": ["a", "b", "c"]})
    summary = get_summary(df)
    assert "shape" in summary
    assert "missing_values" in summary
