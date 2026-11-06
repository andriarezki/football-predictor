"""Download Premier League results (football-data.co.uk) and load them into SQLite.

The raw CSVs are cached in data/raw and are NOT committed to the repo: the
source asks to be credited, and re-downloading keeps this repo small.
"""
from __future__ import annotations

import sqlite3
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425"]
LIVE_SEASONS = SEASONS + ["2526", "2627"]  # used by live.py (the README backtest keeps SEASONS)
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
URL = "https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
UA = "football-predictor/1.0 (portfolio project; github.com/andriarezki)"
KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "AvgH", "AvgD", "AvgA"]


def download(seasons=SEASONS, pause: float = 1.0, refresh: tuple = ()) -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seasons:
        p = RAW / f"E0_{s}.csv"
        if not p.exists() or s in refresh:
            req = urllib.request.Request(URL.format(s=s), headers={"User-Agent": UA})
            p.write_bytes(urllib.request.urlopen(req, timeout=30).read())
            time.sleep(pause)  # be polite
        paths.append(p)
    return paths


def parse(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig")[KEEP].dropna(subset=["HomeTeam", "FTHG", "FTAG"])
        df["season"] = p.stem.split("_")[1]
        frames.append(df)
    m = pd.concat(frames, ignore_index=True)
    m["date"] = pd.to_datetime(m["Date"], dayfirst=True, format="mixed")
    m = m.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag", "FTR": "result",
                          "AvgH": "odds_h", "AvgD": "odds_d", "AvgA": "odds_a"})
    m = m.drop(columns="Date").sort_values("date").reset_index(drop=True)
    m["hg"], m["ag"] = m["hg"].astype(int), m["ag"].astype(int)
    m["match_id"] = m.index + 1
    return m[["match_id", "season", "date", "home", "away", "hg", "ag", "result", "odds_h", "odds_d", "odds_a"]]


def to_sqlite(matches: pd.DataFrame, path: str | Path) -> None:
    df = matches.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(path) as con:
        df.to_sql("matches", con, if_exists="replace", index=False)


def load_matches() -> pd.DataFrame:
    return parse(download())


def fetch_fixtures() -> pd.DataFrame:
    """Upcoming matches (football-data.co.uk fixtures.csv), Premier League only."""
    req = urllib.request.Request(FIXTURES_URL, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig")
    from io import StringIO
    df = pd.read_csv(StringIO(raw))
    df = df[df["Div"] == "E0"].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    cols = {"HomeTeam": "home", "AwayTeam": "away"}
    for c in ("AvgH", "AvgD", "AvgA"):
        if c not in df.columns:
            df[c] = pd.NA
    df = df.rename(columns={**cols, "AvgH": "odds_h", "AvgD": "odds_d", "AvgA": "odds_a"})
    return df[["date", "Time", "home", "away", "odds_h", "odds_d", "odds_a"]].sort_values(["date", "Time"]).reset_index(drop=True)
