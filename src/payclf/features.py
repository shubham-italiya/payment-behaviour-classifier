"""Build the model's input table with SQL (sql/features.sql)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .paths import DB_FILE, SQL_DIR

TARGET = "paid_late"
# Behaviour only: demographic columns (sex, age, marriage, education) are deliberately left out.
FEATURES = [
    "credit_limit", "latest_status", "worst_status", "avg_status", "months_delayed", "months_delayed_last3",
    "latest_bill", "avg_bill", "bill_change_6m", "latest_utilisation", "avg_utilisation",
    "total_paid", "latest_paid", "months_unpaid", "paid_to_billed_ratio",
]


def load_features(db_path: Path = DB_FILE) -> pd.DataFrame:
    query = (SQL_DIR / "features.sql").read_text()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
