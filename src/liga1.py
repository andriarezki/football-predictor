"""Liga 1 Indonesia (BRI Super League): results from Wikipedia, upcoming fixtures from the league site.

Model: the same time-decayed Poisson attack/defence model used for the Premier League. There are no
bookmaker odds here, so the benchmark is the home/draw/away split of last season.
Results on Wikipedia carry no dates, so each season gets one pseudo date and the time decay only
separates "this season" from "last season".
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .poisson import PoissonModel

UA = "football-predictor/1.0 (portfolio project; github.com/andriarezki)"
WIKI = "https://en.wikipedia.org/w/index.php?title={t}&action=raw"
SEASONS = {"2025-26": "2025–26 Super League (Indonesia)", "2026-27": "2026–27 Super League (Indonesia)"}
PSEUDO_DATE = {"2025-26": "2026-03-01", "2026-27": "2026-10-15"}
FIXTURES_URL = "https://ileague.id/fixtures/index/BRI_SUPER_LEAGUE_2026-27?lang=en"


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=40).read().decode("utf-8", errors="ignore")


def parse_wikitext(raw: str) -> tuple[dict, pd.DataFrame]:
    names = {}
    for code, val in re.findall(r"\|name_([A-Z]{3})\s*=\s*(.+)", raw):
        m = re.findall(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", val)
        names[code] = (m[-1] if m else re.sub(r"\{\{[^}]*\}\}|[{}]", "", val)).strip()
    rows = [{"home": names[h], "away": names[a], "hg": int(x), "ag": int(y)}
            for h, a, x, y in re.findall(r"\|match_([A-Z]{3})_([A-Z]{3})\s*=\s*(\d+)[–-](\d+)", raw)
            if h in names and a in names]
    return names, pd.DataFrame(rows)


def load_results() -> dict[str, pd.DataFrame]:
    out = {}
    for season, title in SEASONS.items():
        _, df = parse_wikitext(_get(WIKI.format(t=urllib.parse.quote(title.replace(" ", "_")))))
        df["date"] = pd.Timestamp(PSEUDO_DATE[season])
        out[season] = df
        time.sleep(1)
    return out


def _norm(s: str) -> str:
    s = s.lower().replace("f.c.", "").replace(" fc", "")
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def parse_fixtures(html: str, known_teams: list[str]) -> pd.DataFrame:
    """Rows: date, time (WIB), home, away, venue. Team names are mapped to the Wikipedia names."""
    def map_team(raw: str):
        n = _norm(raw)
        hits = [t for t in known_teams if _norm(t) in n or n in _norm(t)]
        return max(hits, key=len) if hits else None

    rows, date = [], None
    for block in re.split(r'(?=<div align="left" style="margin-top:10px")', html):
        m = re.search(r'margin-top:10px"[^>]*>\s*\w+, (\d{1,2} \w+ \d{4})\s*</div>', block)
        if not m:
            continue
        date = datetime.strptime(m.group(1), "%d %B %Y").date()
        for row in re.findall(r"<tr height=\"25\".*?</tr>", block, flags=re.S):
            teams = re.findall(r"<b>([^<]+)</b>", row)
            t = re.search(r">(\d{2}:\d{2})<", row)
            venue = re.findall(r'width="30%" align="left">\s*([^<]+?)\s*</td>', row)
            if len(teams) == 2 and t:
                h, a = map_team(teams[0]), map_team(teams[1])
                if h and a:
                    rows.append({"date": pd.Timestamp(date), "time": t.group(1), "home": h, "away": a,
                                 "venue": venue[0] if venue else ""})
    return pd.DataFrame(rows)


def fetch_fixtures(known_teams: list[str]) -> pd.DataFrame:
    return parse_fixtures(_get(FIXTURES_URL), known_teams)


def predict(results: dict[str, pd.DataFrame], fixtures: pd.DataFrame) -> pd.DataFrame:
    allm = pd.concat(results.values(), ignore_index=True)
    model = PoissonModel().fit(allm, as_of=pd.Timestamp(PSEUDO_DATE["2026-27"]) + pd.Timedelta(days=21))
    rows = []
    for f in fixtures.itertuples():
        p = model.predict_proba(f.home, f.away)
        rows.append({"date": f.date.strftime("%Y-%m-%d"), "time": f.time + " WIB", "home": f.home, "away": f.away,
                     "venue": f.venue, "p": list(p), "known": f.home in model.teams and f.away in model.teams})
    return pd.DataFrame(rows)


def track_record(results: dict[str, pd.DataFrame]) -> dict:
    """Out-of-sample: train on last season only, score every 2026-27 match whose two teams played last season."""
    prev, cur = results["2025-26"], results["2026-27"].copy()
    model = PoissonModel().fit(prev, as_of=pd.Timestamp(PSEUDO_DATE["2025-26"]))
    cur = cur[cur["home"].isin(model.teams) & cur["away"].isin(model.teams)].reset_index(drop=True)
    if len(cur) < 10:
        return {"n": 0}
    y = np.where(cur["hg"] > cur["ag"], 0, np.where(cur["hg"] == cur["ag"], 1, 2))
    P = np.array([model.predict_proba(h, a) for h, a in zip(cur["home"], cur["away"])])
    freq = np.bincount(np.where(prev["hg"] > prev["ag"], 0, np.where(prev["hg"] == prev["ag"], 1, 2)), minlength=3) / len(prev)
    B = np.tile(freq, (len(cur), 1))
    onehot = np.eye(3)[y]
    def row(M):
        return {"accuracy": float((M.argmax(1) == y).mean()), "brier": float(np.mean(np.sum((M - onehot) ** 2, axis=1))),
                "log_loss": float(-np.mean(np.log(np.clip(M[np.arange(len(y)), y], 1e-9, 1))))}
    return {"n": int(len(cur)), "skipped": int(len(results["2026-27"]) - len(cur)),
            "models": {"poisson (trained on 2025-26)": row(P), "baseline (last season's split)": row(B)}}
