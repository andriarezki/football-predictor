from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "analysis.sql"


def run_queries(con: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    parts = re.split(r"^-- name: (\w+)\s*$", SQL_FILE.read_text(), flags=re.M)
    return {name: pd.read_sql_query(body, con) for name, body in zip(parts[1::2], parts[2::2])}
