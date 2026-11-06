import sqlite3, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if (ROOT / ".deps").exists(): sys.path.insert(0, str(ROOT / ".deps"))
import pandas as pd
from src import analysis, dashboard
bt = pd.read_pickle(ROOT / "data" / "backtest.pkl")
summary = pd.read_csv(ROOT / "docs" / "results.csv")
with sqlite3.connect(ROOT / "data" / "football.db") as con:
    q = analysis.run_queries(con)
dashboard.build(summary, bt, q, ROOT / "docs" / "dashboard.html")
print("ok")
