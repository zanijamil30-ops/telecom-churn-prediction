import pickle
import pandas as pd
from pathlib import Path

def save_model(model, feature_names, path: str = "models/best_model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": list(feature_names)}, f)
    return path

def load_model(path: str = "models/best_model.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]

def align_columns(df: pd.DataFrame, feature_names):
    # ensure same columns / order as training
    return df.reindex(columns=feature_names, fill_value=0)

def predict_df(model, df_aligned: pd.DataFrame):
    preds = model.predict(df_aligned)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_aligned)[:, 1]
    return preds, proba

