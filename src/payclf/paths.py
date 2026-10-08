"""Project paths, resolved from this file so commands work from any folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
RAW_FILE = DATA_DIR / "default_of_credit_card_clients.xls"
DB_FILE = DATA_DIR / "payments.db"
