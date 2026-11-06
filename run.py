"""End-to-end: download -> SQLite -> SQL analysis -> features -> backtest -> dashboard data."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps")) if (ROOT / ".deps").exists() else None

from src import analysis, data, evaluate, features  # noqa: E402


def main():
    matches = data.load_matches()
    db = ROOT / "data" / "football.db"
    data.to_sqlite(matches, db)
    with sqlite3.connect(db) as con:
        q = analysis.run_queries(con)
    feat = features.build(matches)
    bt = evaluate.backtest(feat)
    summary = evaluate.summarise(bt)
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "results.csv", index=False)
    bt.to_pickle(ROOT / "data" / "backtest.pkl")
    for name, df in q.items():
        df.to_csv(out / f"sql_{name}.csv", index=False)
    print(summary.to_string(index=False))
    print(len(matches), "matches;", summary["matches"].max(), "backtested")


if __name__ == "__main__":
    main()
