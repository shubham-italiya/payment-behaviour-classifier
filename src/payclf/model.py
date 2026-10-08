"""Candidate models, threshold tuning and evaluation."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def candidates(seed: int = SEED) -> dict:
    """Two model families x two ways of handling the 3.5:1 class imbalance."""
    def gbm(**kw):
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=seed, **kw)

    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=2000))]),
        "logistic_regression_smote": ImbPipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=seed)),
                                                  ("model", LogisticRegression(max_iter=2000))]),
        "gradient_boosting_class_weights": gbm(class_weight="balanced"),
        "gradient_boosting_smote": ImbPipeline([("smote", SMOTE(random_state=seed)), ("model", gbm())]),
    }


def folds(seed: int = SEED, n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def with_tuned_threshold(estimator, seed: int = SEED) -> TunedThresholdClassifierCV:
    """Choose the decision threshold that maximises F1, using cross-validation on the training data only."""
    return TunedThresholdClassifierCV(estimator, scoring="f1", cv=folds(seed), random_state=seed)


def evaluate(model, X, y, threshold: float | None = None) -> dict:
    """Scores on held-out data. threshold=None uses the model's own decision rule."""
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X) if threshold is None else (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
