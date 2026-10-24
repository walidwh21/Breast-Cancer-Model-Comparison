"""End-to-end experiment: load -> inspect -> tune -> evaluate -> plot.

Usage:  python run_experiment.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from tumorclf import data, evaluation, models, plots
from tumorclf.config import REPORT_DIR, Settings


def main():
    cfg = Settings()
    frame = data.load_frame()
    print("Data check:", data.quality_report(frame))
    X_train, X_test, y_train, y_test = data.split(frame, cfg)

    plots.class_balance(frame)
    plots.feature_separation(frame)
    plots.redundancy_clusters(frame)

    fitted, rows, cv_rows = {}, [], []
    cv = RepeatedStratifiedKFold(n_splits=cfg.cv_folds, n_repeats=cfg.cv_repeats, random_state=cfg.seed + 1)
    for name, (pipe, space) in models.build_candidates(cfg.seed).items():
        search = models.tune(name, pipe, space, X_train, y_train, cfg)
        fitted[name] = search.best_estimator_
        for s in cross_val_score(search.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc"):
            cv_rows.append({"model": name, "roc_auc": s})
        rows.append({"model": name, "best_params": str(search.best_params_),
                     **evaluation.score_model(search.best_estimator_, X_test, y_test)})
        print(f"{name:22s} tuned (search AUC {search.best_score_:.4f})")

    results = pd.DataFrame(rows).sort_values(["recall", "roc_auc"], ascending=False)
    print(results.drop(columns="best_params").round(3).to_string(index=False))

    plots.cv_boxplot(pd.DataFrame(cv_rows))
    plots.roc_overlay(fitted, X_test, y_test)
    plots.confusion_panel(fitted, X_test, y_test)
    champion = results.iloc[0]["model"]
    plots.importance_bar(evaluation.top_features(fitted[champion], X_test, y_test, cfg.seed), champion)
    plots.scoreboard(results)

    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "holdout_results.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps({"champion": champion, "settings": cfg.__dict__}, indent=2))
    print("Champion:", champion)


if __name__ == "__main__":
    main()
