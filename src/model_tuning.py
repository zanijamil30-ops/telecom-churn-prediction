from __future__ import annotations
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger("tuning")

def tune_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    param_grid = [
        {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100]},
        {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.01, 0.1, 1, 10, 100]},
        {"solver": ["saga"], "penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100]},
    ]
    grid = GridSearchCV(log_reg, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    logger.info(f"Best Logistic Regression params: {grid.best_params_} | score={grid.best_score_:.4f}")
    return grid.best_estimator_

def build_soft_voting():
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)
    svm = SVC(probability=True, kernel="rbf", C=1, gamma="scale", random_state=42)
    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("svm", svm)], voting="soft")
    return ensemble

def build_stacking():
    base = [
        ("rf", RandomForestClassifier(n_estimators=150, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ]
    meta = LogisticRegression(max_iter=1000)
    stack = StackingClassifier(estimators=base, final_estimator=meta, passthrough=True)
    return stack

def fit_and_select_best(X_train, y_train, X_test, y_test):
    # Tune LR
    best_lr = tune_logistic_regression(X_train, y_train)

    # Soft voting
    voting = build_soft_voting()
    voting.fit(X_train, y_train)
    voting_acc = accuracy_score(y_test, voting.predict(X_test))
    logger.info(f"Soft Voting accuracy: {voting_acc:.4f}")

    # Stacking (often best in your script)
    stacking = build_stacking()
    stacking.fit(X_train, y_train)
    stacking_acc = accuracy_score(y_test, stacking.predict(X_test))
    logger.info(f"Stacking accuracy: {stacking_acc:.4f}")

    # Choose best by accuracy
    best_model, best_name, best_acc = max(
        [(best_lr, "Tuned LogisticRegression", accuracy_score(y_test, best_lr.predict(X_test))),
         (voting, "Soft Voting Ensemble", voting_acc),
         (stacking, "Stacking Ensemble", stacking_acc)],
        key=lambda t: t[2]
    )

    logger.info(f"Selected best model: {best_name} (acc={best_acc:.4f})")
    return best_model, {"name": best_name, "accuracy": best_acc}

