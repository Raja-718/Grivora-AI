# model.py - Train and evaluate ML models
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def train_model(df: pd.DataFrame, target_col: str, task: str = "classification"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
        metric = f"Accuracy: {score:.4f}"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        score = mean_squared_error(y_test, model.predict(X_test), squared=False)
        metric = f"RMSE: {score:.4f}"

    with open("models/model_v1.pkl", "wb") as f:
        pickle.dump(model, f)

    return {"metric": metric, "features": list(X.columns)}
