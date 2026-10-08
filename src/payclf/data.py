"""Download the UCI dataset and load it into SQLite tables: customers, statements and outcomes.

Dataset: "Default of Credit Card Clients" (Yeh, 2009), UCI Machine Learning Repository,
https://doi.org/10.24432/C55S3H, licensed CC BY 4.0. The file is not stored in this repository.
"""
from __future__ import annotations

import sqlite3
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .paths import DB_FILE, RAW_FILE, SQL_DIR

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
# Column suffix -> month_index (1 = September 2005, the most recent month). The file has no PAY_1.
STATUS_COLUMNS = {1: "PAY_0", 2: "PAY_2", 3: "PAY_3", 4: "PAY_4", 5: "PAY_5", 6: "PAY_6"}
TARGET_COLUMN = "default payment next month"


def download(dest: Path = RAW_FILE) -> Path:
    """Download and unzip the .xls file once; later calls reuse it."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "uci.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            name = next(n for n in z.namelist() if n.endswith(".xls"))
            dest.write_bytes(z.read(name))
    return dest


def read_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """The sheet has a title row above the real header."""
    return pd.read_excel(path, header=1)


def to_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reshape one wide row per customer into the three tables of sql/schema.sql."""
    customers = pd.DataFrame({
        "customer_id": raw["ID"].astype(int),
        "credit_limit": raw["LIMIT_BAL"].astype(float),
        "sex": raw["SEX"], "education": raw["EDUCATION"], "marriage": raw["MARRIAGE"], "age": raw["AGE"],
    })
    months = []
    for month, status_col in STATUS_COLUMNS.items():
        months.append(pd.DataFrame({
            "customer_id": raw["ID"].astype(int),
            "month_index": month,
            "bill_amount": raw[f"BILL_AMT{month}"].astype(float),
            "paid_amount": raw[f"PAY_AMT{month}"].astype(float),
            "repayment_status": raw[status_col].astype(int),
        }))
    statements = pd.concat(months, ignore_index=True).sort_values(["customer_id", "month_index"])
    outcomes = pd.DataFrame({"customer_id": raw["ID"].astype(int), "paid_late": raw[TARGET_COLUMN].astype(int)})
    return customers, statements.reset_index(drop=True), outcomes


def build_database(raw: pd.DataFrame, db_path: Path = DB_FILE) -> Path:
    """Create a fresh SQLite database with the schema and the reshaped data."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    customers, statements, outcomes = to_tables(raw)
    with sqlite3.connect(db_path) as conn:
        conn.executescript((SQL_DIR / "schema.sql").read_text())
        customers.to_sql("customers", conn, if_exists="append", index=False)
        statements.to_sql("statements", conn, if_exists="append", index=False)
        outcomes.to_sql("outcomes", conn, if_exists="append", index=False)
    return db_path


def main() -> None:
    raw = read_raw(download())
    path = build_database(raw)
    with sqlite3.connect(path) as conn:
        for table in ("customers", "statements", "outcomes"):
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            print(f"{table:<11} {count:>7,} rows")
    print(f"Database ready: {path}")


if __name__ == "__main__":
    main()
