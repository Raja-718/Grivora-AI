# tools.py - Python functions available as tools for agents

from src.data_loader import load_file
from src.preprocessor import clean_data
from src.visualizer import generate_chart
from src.predictor import run_prediction

def tool_load_data(file_path: str) -> str:
    df = load_file(file_path)
    return df.head(20).to_string()

def tool_clean_data(file_path: str) -> str:
    df = load_file(file_path)
    cleaned = clean_data(df)
    return cleaned.head(20).to_string()

def tool_predict(file_path: str, target_column: str) -> str:
    result = run_prediction(file_path, target_column)
    return str(result)

def tool_generate_chart(file_path: str, chart_type: str = "bar") -> str:
    df = load_file(file_path)
    path = generate_chart(df, chart_type)
    return f"Chart saved at: {path}"
