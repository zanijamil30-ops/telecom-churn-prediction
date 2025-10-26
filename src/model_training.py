from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger("training")

def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }
    return models

def train_baselines(X_train, y_train):
    models = build_models()
    for name, m in models.items():
        m.fit(X_train, y_train)
        logger.info(f"Trained {name}")
    return models

