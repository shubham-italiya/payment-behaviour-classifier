"""Run the full experiment and write reports/:  python -m payclf.train

1. Build features with SQL and hold out 20% of customers as a test set (never used for choices).
2. Compare four candidates with 5-fold cross-validation on the training set (PR-AUC, ROC-AUC).
3. Take the best one, tune its decision threshold for F1 with cross-validation, fit on all training data.
4. Score it once on the test set, next to a plain logistic-regression baseline at threshold 0.5.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay  # noqa: E402
from sklearn.model_selection import cross_validate, train_test_split  # noqa: E402

from .features import FEATURES, TARGET, load_features  # noqa: E402
from .model import SEED, candidates, evaluate, folds, with_tuned_threshold  # noqa: E402
from .paths import DB_FILE, REPORTS_DIR  # noqa: E402


def main() -> None:
    if not DB_FILE.exists():
        raise SystemExit("No database yet. Run:  python -m payclf.data")
    started = time.time()
    df = load_features()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"{len(df):,} customers, {y.mean():.1%} paid late. Train {len(X_train):,} / test {len(X_test):,}.")

    # 2. Model comparison on the training set only.
    comparison = []
    for name, est in candidates().items():
        cv = cross_validate(est, X_train, y_train, cv=folds(), scoring=["average_precision", "roc_auc"], n_jobs=-1)
        comparison.append({"model": name,
                           "cv_pr_auc": cv["test_average_precision"].mean(),
                           "cv_pr_auc_std": cv["test_average_precision"].std(),
                           "cv_roc_auc": cv["test_roc_auc"].mean()})
        print(f"  {name:<34} PR-AUC {comparison[-1]['cv_pr_auc']:.3f} ± {comparison[-1]['cv_pr_auc_std']:.3f}"
              f"   ROC-AUC {comparison[-1]['cv_roc_auc']:.3f}")
    best_name = max(comparison, key=lambda r: r["cv_pr_auc"])["model"]

    # 3. Best model with a tuned threshold; 4. one look at the test set.
    tuned = with_tuned_threshold(candidates()[best_name]).fit(X_train, y_train)
    baseline = candidates()["logistic_regression"].fit(X_train, y_train)
    results = {"baseline (logistic regression, threshold 0.50)": evaluate(baseline, X_test, y_test),
               f"best ({best_name}, threshold {tuned.best_threshold_:.2f})": evaluate(tuned, X_test, y_test)}

    REPORTS_DIR.mkdir(exist_ok=True)
    metrics = {"n_customers": len(df), "positive_rate": round(float(y.mean()), 4), "test_size": len(X_test),
               "cv_comparison": comparison, "best_model": best_name,
               "threshold": round(float(tuned.best_threshold_), 4), "test_results": results}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_markdown(metrics)
    plot_pr_curves(baseline, tuned, X_test, y_test, best_name)
    plot_confusion(tuned, X_test, y_test)
    plot_importance(tuned, X_test, y_test)
    for label, r in results.items():
        print(f"TEST {label}: F1 {r['f1']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"PR-AUC {r['pr_auc']:.3f}  ROC-AUC {r['roc_auc']:.3f}")
    print(f"Done in {time.time() - started:.0f}s. Reports in {REPORTS_DIR}")


def write_markdown(m: dict) -> None:
    lines = ["## Model comparison (5-fold cross-validation on the training set)", "",
             "| Model | PR-AUC | ROC-AUC |", "|---|---|---|"]
    for r in sorted(m["cv_comparison"], key=lambda r: -r["cv_pr_auc"]):
        lines.append(f"| {r['model']} | {r['cv_pr_auc']:.3f} ± {r['cv_pr_auc_std']:.3f} | {r['cv_roc_auc']:.3f} |")
    lines += ["", f"## Held-out test set ({m['test_size']:,} customers, {m['positive_rate']:.1%} paid late)", "",
              "| Model | F1 | Precision | Recall | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|"]
    for label, r in m["test_results"].items():
        lines.append(f"| {label} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['pr_auc']:.3f} | {r['roc_auc']:.3f} |")
    (REPORTS_DIR / "results.md").write_text("\n".join(lines) + "\n")


def plot_pr_curves(baseline, tuned, X, y, best_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    PrecisionRecallDisplay.from_estimator(baseline, X, y, name="Logistic regression (baseline)", ax=ax)
    nice = best_name.replace("_class_weights", " (class weights)").replace("_smote", " (SMOTE)").replace("_", " ")
    PrecisionRecallDisplay.from_estimator(tuned, X, y, name=nice.capitalize(), ax=ax)
    ax.axhline(y.mean(), ls="--", c="grey", lw=1, label=f"Random ({y.mean():.0%} positive)")
    ax.set_title("Precision-recall on the held-out test set")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)


def plot_confusion(tuned, X, y) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, tuned.predict(X), display_labels=["On time", "Late"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Test set, threshold {tuned.best_threshold_:.2f}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_importance(tuned, X, y) -> None:
    """Permutation importance: how much PR-AUC drops when one feature is shuffled."""
    imp = permutation_importance(tuned, X, y, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=-1)
    s = pd.Series(imp.importances_mean, index=X.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    s.plot.barh(ax=ax, color="#2563eb")
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title("Top 10 features (permutation importance, test set)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
