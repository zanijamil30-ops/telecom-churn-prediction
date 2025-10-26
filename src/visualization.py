import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def plot_churn_distribution(df: pd.DataFrame, target_col: str = "Churn", save_to: str | None = None):
    ax = df[target_col].value_counts().plot(kind="bar")
    ax.set_title("Churn distribution (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    fig = ax.get_figure()
    if save_to:
        fig.savefig(save_to, bbox_inches="tight")
    plt.show()

def boxplots_by_churn(df: pd.DataFrame, cols=None, target_col: str = "Churn"):
    cols = cols or ["Tenure", "MonthlyCharges", "TotalCharges", "Age"]
    for col in cols:
        if col in df.columns:
            df.boxplot(column=col, by=target_col, grid=False)
            plt.suptitle("")
            plt.title(f"{col} by Churn")
            plt.xlabel("Churn")
            plt.show()

def churn_rate_by_category(df: pd.DataFrame, cats=None, target_col: str = "Churn"):
    cats = cats or ["ContractType", "InternetService", "PaymentMethod", "Gender"]
    for cat in cats:
        if cat in df.columns:
            rates = df.groupby(cat)[target_col].mean().sort_values(ascending=False)
            ax = rates.plot(kind="bar")
            ax.set_title(f"Churn rate by {cat}")
            ax.set_ylabel("Churn rate")
            plt.show()

def top_feature_importances(X_train, y_train, top_n=15):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(top_n)
    ax = imp.plot(kind="bar")
    ax.set_title("Top Feature Importances (RandomForest)")
    ax.set_ylabel("Importance")
    plt.show()
    return imp

