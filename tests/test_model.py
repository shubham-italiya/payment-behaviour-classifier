import pytest
from sklearn.datasets import make_classification

from payclf.model import candidates, evaluate, with_tuned_threshold


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=8, weights=[0.78], random_state=0)
    return X, y


@pytest.mark.parametrize("name", list(candidates()))
def test_every_candidate_trains_and_predicts(name, data):
    X, y = data
    model = candidates()[name].fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),) and ((proba >= 0) & (proba <= 1)).all()


def test_threshold_is_tuned_and_scores_are_valid(data):
    X, y = data
    tuned = with_tuned_threshold(candidates()["logistic_regression"]).fit(X, y)
    assert 0 < tuned.best_threshold_ < 1
    scores = evaluate(tuned, X, y)
    assert 0 <= scores["f1"] <= 1 and 0.5 < scores["roc_auc"] <= 1
    assert sum(scores["confusion_matrix"].values()) == len(y)
