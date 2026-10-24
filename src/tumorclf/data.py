"""Data loading and splitting.

Convention used throughout this project: the *positive class is malignant*
(label 1). scikit-learn's bundled data encodes malignant as 0, so labels are
flipped once, here, to keep recall/precision interpretable downstream.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from .config import Settings

CLASS_NAMES = ("benign", "malignant")


def load_frame() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    frame["malignant"] = 1 - frame.pop("target")
    return frame


def split(frame: pd.DataFrame, cfg: Settings):
    features = frame.drop(columns="malignant")
    labels = frame["malignant"]
    return train_test_split(
        features,
        labels,
        test_size=cfg.test_fraction,
        stratify=labels,
        random_state=cfg.seed,
    )


def quality_report(frame: pd.DataFrame) -> dict:
    """Cheap sanity checks printed before any modelling."""
    return {
        "rows": len(frame),
        "features": frame.shape[1] - 1,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "malignant_share": round(float(frame["malignant"].mean()), 3),
    }
