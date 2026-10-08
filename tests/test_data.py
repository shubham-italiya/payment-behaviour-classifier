from payclf.data import to_tables


def test_reshape_gives_six_statements_per_customer(raw):
    customers, statements, outcomes = to_tables(raw)
    assert len(customers) == 2 and len(outcomes) == 2
    assert len(statements) == 12
    assert set(statements["month_index"]) == {1, 2, 3, 4, 5, 6}


def test_month_one_is_the_most_recent_month(raw):
    _, statements, _ = to_tables(raw)
    latest = statements.query("customer_id == 2 and month_index == 1").iloc[0]
    assert latest["repayment_status"] == 2  # PAY_0 in the raw file
    assert latest["bill_amount"] == 19_000


def test_outcome_is_copied(raw):
    _, _, outcomes = to_tables(raw)
    assert outcomes.set_index("customer_id")["paid_late"].to_dict() == {1: 0, 2: 1}
