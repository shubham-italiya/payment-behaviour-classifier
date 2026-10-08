import pytest

from payclf.data import build_database
from payclf.features import FEATURES, load_features


@pytest.fixture
def features(raw, tmp_path):
    return load_features(build_database(raw, tmp_path / "test.db")).set_index("customer_id")


def test_one_row_per_customer_with_all_features(features):
    assert len(features) == 2
    assert set(FEATURES) <= set(features.columns)
    assert not features[FEATURES].isna().any().any()


def test_delay_counts(features):
    assert features.loc[1, "months_delayed"] == 0
    assert features.loc[2, "months_delayed"] == 3
    assert features.loc[2, "months_delayed_last3"] == 3
    assert features.loc[2, "worst_status"] == 2


def test_payment_ratios(features):
    assert features.loc[1, "paid_to_billed_ratio"] == pytest.approx(1.0)
    assert features.loc[2, "paid_to_billed_ratio"] == pytest.approx(3500 / 82_000)
    assert features.loc[2, "latest_utilisation"] == pytest.approx(19_000 / 20_000)
    assert features.loc[2, "months_unpaid"] == 2


def test_demographics_are_not_features():
    assert not {"sex", "age", "marriage", "education"} & set(FEATURES)
