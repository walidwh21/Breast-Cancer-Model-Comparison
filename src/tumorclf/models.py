"""Model zoo and hyper-parameter search spaces.

Every estimator is wrapped in a Pipeline so scaling is fitted on training
folds only (no leakage during cross-validation).
"""
from __future__ import annotations

from scipy.stats import loguniform, randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import Settings


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def build_candidates(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, search space)."""
    return {
        "Logistic Regression": (
            _pipe(LogisticRegression(max_iter=5000, random_state=seed)),
            {"clf__C": loguniform(1e-3, 1e2)},
        ),
        "RBF SVM": (
            _pipe(CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), cv=3)),
            {"clf__estimator__C": loguniform(1e-1, 1e2),
             "clf__estimator__gamma": loguniform(1e-4, 1e-1)},
        ),
        "Random Forest": (
            _pipe(RandomForestClassifier(random_state=seed, n_jobs=-1)),
            {
                "clf__n_estimators": randint(100, 400),
                "clf__max_depth": [None, 4, 8, 12],
                "clf__min_samples_leaf": randint(1, 6),
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        "Gradient Boosting": (
            _pipe(GradientBoostingClassifier(random_state=seed)),
            {
                "clf__n_estimators": randint(80, 300),
                "clf__learning_rate": loguniform(0.02, 0.3),
                "clf__max_depth": randint(2, 5),
                "clf__subsample": [0.7, 0.85, 1.0],
            },
        ),
        "K-Nearest Neighbors": (
            _pipe(KNeighborsClassifier()),
            {
                "clf__n_neighbors": randint(3, 25),
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Decision Tree": (
            _pipe(DecisionTreeClassifier(random_state=seed)),
            {
                "clf__max_depth": randint(2, 9),
                "clf__min_samples_leaf": randint(1, 10),
                "clf__criterion": ["gini", "entropy"],
            },
        ),
    }


def tune(name: str, pipe: Pipeline, space: dict, X, y, cfg: Settings) -> RandomizedSearchCV:
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.cv_folds, n_repeats=cfg.cv_repeats, random_state=cfg.seed
    )
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=cfg.search_iterations,
        scoring="roc_auc",
        cv=cv,
        random_state=cfg.seed,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X, y)
