import pandas as pd
import pytest


@pytest.fixture
def raw() -> pd.DataFrame:
    """Two customers in the UCI file's wide format. Customer 1 pays on time; customer 2 falls behind."""
    rows = []
    for cid, limit, statuses, bills, paid, late in (
        (1, 50_000, [-1, -1, 0, 0, 0, 0], [1000, 900, 800, 700, 600, 500], [1000, 900, 800, 700, 600, 500], 0),
        (2, 20_000, [2, 1, 1, 0, 0, 0], [19_000, 18_000, 15_000, 12_000, 10_000, 8_000],
         [0, 0, 500, 1000, 1000, 1000], 1),
    ):
        row = {"ID": cid, "LIMIT_BAL": limit, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
               "default payment next month": late}
        for month, col in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], start=1):
            row[col] = statuses[month - 1]
            row[f"BILL_AMT{month}"] = bills[month - 1]
            row[f"PAY_AMT{month}"] = paid[month - 1]
        rows.append(row)
    return pd.DataFrame(rows)
