# predictor.py - Load saved model and run predictions
import pickle
import pandas as pd
from src.data_loader import load_file
from src.preprocessor import clean_data, encode_categoricals

def load_model(model_path: str = "models/model_v1.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def run_prediction(file_path: str, target_column: str = None) -> list:
    df = load_file(file_path)
    df = clean_data(df)
    df = encode_categoricals(df)
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    model = load_model()
    predictions = model.predict(df)
    return predictions.tolist()
