"""Held-out evaluation helpers."""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_model(fitted, X_test, y_test) -> dict:
    pred = fitted.predict(X_test)
    prob = fitted.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "balanced_acc": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def top_features(fitted, X_test, y_test, seed: int, k: int = 10) -> pd.Series:
    """Model-agnostic importance: drop in ROC-AUC when a column is shuffled."""
    res = permutation_importance(
        fitted, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=seed, n_jobs=-1
    )
    return pd.Series(res.importances_mean, index=X_test.columns).nlargest(k)[::-1]
