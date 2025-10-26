from __future__ import annotations
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger("preprocessing")

def _read_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_raw_dataframe(params_path: str = "params.yaml") -> pd.DataFrame:
    p = _read_params(params_path)
    raw_path = Path(p["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path.resolve()}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded raw data: {df.shape} from {raw_path}")
    return df

def preprocess_dataframe(df: pd.DataFrame, params_path: str = "params.yaml"):
    p = _read_params(params_path)
    target = p["data"]["target_column"]

    # Drop columns
    drop_cols = [c for c in p["preprocessing"]["drop_columns"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Handle TotalCharges missing / spaces -> NaN -> median
    if "handle_missing" in p["preprocessing"]:
        miss_col = p["preprocessing"]["handle_missing"]["column"]
        if miss_col in df.columns:
            df[miss_col] = df[miss_col].replace(" ", np.nan).astype(float)
            if df[miss_col].isna().any():
                strategy = p["preprocessing"]["handle_missing"].get("strategy", "median")
                if strategy == "median":
                    df[miss_col] = df[miss_col].fillna(df[miss_col].median())
                elif strategy == "mean":
                    df[miss_col] = df[miss_col].fillna(df[miss_col].mean())

    # Binary encode (Yes/No, Male/Female)
    bin_cols = [c for c in p["preprocessing"]["binary_encode"] if c in df.columns]
    for col in bin_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0}).astype(int)

    # One-hot encode
    oh_cols = [c for c in p["preprocessing"]["one_hot_encode"] if c in df.columns]
    df = pd.get_dummies(df, columns=oh_cols, drop_first=True)

    # Scale numeric
    scaler = None
    num_cols = [c for c in p["preprocessing"]["numeric_features"] if c in df.columns]
    if p["preprocessing"].get("scale_numeric", True) and num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Split features/target
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    logger.info(f"Processed data: X={X.shape}, y={y.shape}. Features: {len(X.columns)}")

    return X, y, scaler, X.columns.tolist()

def load_and_preprocess(params_path: str = "params.yaml"):
    p = _read_params(params_path)
    test_size = p["split"]["test_size"]
    rand = p["split"]["random_state"]

    df = load_raw_dataframe(params_path)
    X, y, scaler, feature_names = preprocess_dataframe(df, params_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand
    )

    # save processed data optionally
    proc_path = Path(p["data"]["processed_path"])
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X, y], axis=1).to_csv(proc_path, index=False)

    logger.info(f"Train/Test split: {X_train.shape} / {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, feature_names

