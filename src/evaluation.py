from __future__ import annotations
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.logger import get_logger

logger = get_logger("evaluation")

def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rows.append({"Model": name, "Accuracy": acc})
        logger.info(f"{name} accuracy = {acc:.4f}")
    df = pd.DataFrame(rows)
    df["Z-Score"] = zscore(df["Accuracy"])
    return df

def detailed_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    logger.info(f"\nConfusion Matrix:\n{cm}\n\nReport:\n{report}")
    return cm, report

