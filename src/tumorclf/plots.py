"""Figure factory. Every function saves a PNG and returns its path."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import FIGURE_DIR

PALETTE = {"benign": "#3b82a0", "malignant": "#d1495b"}


def _save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def class_balance(frame):
    counts = frame["malignant"].map({0: "benign", 1: "malignant"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set(title="Class balance", ylabel="samples")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "01_class_balance.png")


def feature_separation(frame, k=8):
    """Which features separate the classes best (absolute point-biserial corr)."""
    corr = frame.corr()["malignant"].drop("malignant").abs().nlargest(k).index
    fig, axes = plt.subplots(2, k // 2, figsize=(12, 5.5))
    for ax, col in zip(axes.ravel(), corr):
        sns.kdeplot(
            data=frame.assign(label=frame["malignant"].map({0: "benign", 1: "malignant"})),
            x=col, hue="label", palette=PALETTE, fill=True, alpha=0.4, ax=ax, legend=False,
        )
        ax.set_title(col, fontsize=10)
        ax.set(xlabel="", ylabel="")
    fig.suptitle("Top class-separating features (density by class)")
    fig.tight_layout()
    return _save(fig, "02_feature_separation.png")


def redundancy_clusters(frame):
    mean_cols = [c for c in frame.columns if c.startswith("mean ")]
    corr = frame[mean_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=True, fmt=".1f",
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Redundancy among the 'mean' features")
    return _save(fig, "03_feature_redundancy.png")


def cv_boxplot(cv_table):
    order = cv_table.groupby("model")["roc_auc"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.boxplot(data=cv_table, y="model", x="roc_auc", order=order, color="#9ec5d6", ax=ax)
    sns.stripplot(data=cv_table, y="model", x="roc_auc", order=order, color="#264653", size=3, ax=ax)
    ax.set(title="Cross-validated ROC-AUC (training data only)", xlabel="ROC-AUC", ylabel="")
    return _save(fig, "04_cv_scores.png")


def roc_overlay(fitted_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title("ROC curves on the held-out set")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "05_roc_curves.png")


def confusion_panel(fitted_models, X_test, y_test):
    n = len(fitted_models)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(11, 6.5))
    for ax, (name, model) in zip(axes.ravel(), fitted_models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=["benign", "malignant"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    return _save(fig, "06_confusion_matrices.png")


def importance_bar(series, model_name):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(series.index, series.values, color="#2a9d8f")
    ax.set(title=f"Permutation importance — {model_name}", xlabel="mean drop in ROC-AUC")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "07_permutation_importance.png")


def scoreboard(results):
    metrics = ["recall", "precision", "f1", "roc_auc"]
    data = results.set_index("model")[metrics]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.85, vmax=1, cbar=False, ax=ax)
    ax.set(title="Held-out scoreboard (malignant = positive class)", ylabel="", xlabel="")
    return _save(fig, "08_scoreboard.png")
