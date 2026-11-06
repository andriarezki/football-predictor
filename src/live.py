"""Live page: predictions for upcoming Premier League matches + honest running track record.

Everything the track record shows is out-of-sample: each month's matches are predicted by models
trained only on earlier matches (the same walk-forward backtest as the README).
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data, evaluate, features

SEASON_START = "2026-08-01"


def predict_fixtures(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    feat = features.build(matches).sort_values(["date", "match_id"])
    train = feat.dropna(subset=features.FEATURES)
    y = train["result"].map({"H": 0, "D": 1, "A": 2})
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(train[features.FEATURES], y)
    rows = []
    for fx in fixtures.itertuples():
        one = pd.DataFrame([{"match_id": int(matches["match_id"].max()) + 1, "season": "live", "date": fx.date,
                             "home": fx.home, "away": fx.away, "hg": 0, "ag": 0, "result": "D",
                             "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan}])
        f = features.build(pd.concat([matches, one], ignore_index=True))
        r = f[f["match_id"] == one["match_id"].iloc[0]]
        X = r[features.FEATURES].fillna(train[features.FEATURES].median())
        p = model.predict_proba(X)[0]
        mk = None
        if pd.notna(fx.odds_h):
            inv = 1 / np.array([fx.odds_h, fx.odds_d, fx.odds_a], dtype=float)
            mk = (inv / inv.sum()).tolist()
        rows.append({"date": fx.date.strftime("%Y-%m-%d"), "time": fx.Time, "home": fx.home, "away": fx.away,
                     "p": p.tolist(), "market": mk})
    return pd.DataFrame(rows)


def track_record(matches: pd.DataFrame) -> dict:
    feat = features.build(matches)
    bt = evaluate.backtest(feat, test_from=SEASON_START)
    if bt.empty:
        return {"n": 0}
    s = evaluate.summarise(bt).set_index("model")
    keep = {k: s.loc[k].to_dict() for k in ("logistic", "poisson", "bookmaker", "baseline_freq") if k in s.index}
    # latest results with the logistic prediction
    lg = bt[bt["model"] == "logistic"].merge(matches[["match_id", "home", "away", "hg", "ag", "date"]], on="match_id", suffixes=("", "_m"))
    lg = lg.sort_values("date").tail(10).iloc[::-1]
    recent = [{"date": r.date.strftime("%Y-%m-%d"), "home": r.home, "away": r.away, "score": f"{r.hg}-{r.ag}",
               "p": [r.pH, r.pD, r.pA], "y": int(r.y)} for r in lg.itertuples()]
    return {"n": int(s["matches"].max()), "models": keep, "recent": recent}


LABEL = {"logistic": "logistic regression", "poisson": "Poisson", "bookmaker": "bookmaker odds", "baseline_freq": "baseline (class split)"}


def _bar(p):
    seg = "".join(f'<span class="s{i}" style="width:{v*100:.1f}%">{v*100:.0f}%</span>' for i, v in enumerate(p))
    return f'<div class="bar">{seg}</div>'


def render_fixtures(fixtures: pd.DataFrame, empty_msg: str) -> str:
    if fixtures is None or fixtures.empty:
        return f'<p class="muted">{empty_msg}</p>'
    out = ""
    for r in fixtures.itertuples():
        pick = ["Home win", "Draw", "Away win"][int(np.argmax(r.p))]
        extra = ""
        if getattr(r, "market", None):
            extra += f'<div class="mk">Market: {" / ".join(f"{v*100:.0f}%" for v in r.market)}</div>'
        if getattr(r, "venue", ""):
            extra += f'<div class="mk">{html.escape(r.venue)}</div>'
        warn = "" if getattr(r, "known", True) else '<div class="mk">A team has no history here, so this is a rough guess.</div>'
        out += (f'<div class="m"><div class="t"><b>{html.escape(r.home)}</b> vs <b>{html.escape(r.away)}</b>'
                f'<span class="d">{r.date} {r.time}</span></div>{_bar(r.p)}<div class="lg"><i>Home</i><i>Draw</i><i>Away</i></div>'
                f'<div class="pk">Leaning: {pick}</div>{extra}{warn}</div>')
    return out


def render_track(tr: dict, intro: str) -> str:
    if not tr.get("n"):
        return '<p class="muted">Not enough matches yet.</p>'
    rows = "".join(f"<tr><td>{LABEL.get(k, k)}</td><td>{v['accuracy']*100:.1f}%</td><td>{v['brier']:.3f}</td><td>{v['log_loss']:.3f}</td></tr>"
                   for k, v in tr["models"].items())
    out = (f'<p class="muted">{intro.format(n=tr["n"])}</p>'
           f'<table><tr><th>model</th><th>accuracy</th><th>Brier</th><th>log loss</th></tr>{rows}</table>')
    if tr.get("recent"):
        rec = ""
        for r in tr["recent"]:
            hit = int(np.argmax(r["p"])) == r["y"]
            rec += (f'<tr><td>{r["date"][5:]}</td><td>{html.escape(r["home"])} {r["score"]} {html.escape(r["away"])}</td>'
                    f'<td>{max(r["p"])*100:.0f}%</td><td class="{"ok" if hit else "no"}">{"✓" if hit else "✗"}</td></tr>')
        out += f'<h3>Last 10 matches</h3><table><tr><th>date</th><th>result</th><th>confidence</th><th></th></tr>{rec}</table>'
    return out


def build_page(leagues: list[dict], updated: str) -> str:
    """leagues: [{'key','name','fixtures_html','track_html','note'}]"""
    radios = "".join(f'<input type="radio" name="lg" id="r-{l["key"]}" {"checked" if i == 0 else ""}>' for i, l in enumerate(leagues))
    labels = "".join(f'<label for="r-{l["key"]}">{l["name"]}</label>' for l in leagues)
    panels = "".join(
        f'<section id="p-{l["key"]}"><h2>Next matches</h2>{l["fixtures_html"]}<h2>Season report card</h2>{l["track_html"]}'
        f'<p class="muted">{l.get("note", "")}</p></section>' for l in leagues)
    show = "".join(f'#r-{l["key"]}:checked~main #p-{l["key"]}{{display:block}}#r-{l["key"]}:checked~main label[for=r-{l["key"]}]{{background:var(--a);color:#fff}}' for l in leagues)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Football match predictions</title>
<style>
:root{{--bg:#f7f8f6;--fg:#1d2733;--card:#fff;--line:#e1e5e9;--mut:#52606d;--a:#2c7a7b;--b:#9aa5b1;--c:#d98e04}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181c;--fg:#e8ecef;--card:#1d2328;--line:#2d353c;--mut:#9aa7b2}}}}
body{{font-family:system-ui,Arial,sans-serif;margin:0;background:var(--bg);color:var(--fg)}}main{{max-width:760px;margin:0 auto;padding:20px 16px 48px}}
input[name=lg]{{display:none}}section{{display:none}}{show}
.tabs{{display:flex;gap:8px;margin:12px 0}}.tabs label{{padding:8px 14px;border:1px solid var(--line);border-radius:999px;cursor:pointer;background:var(--card)}}
h1{{font-size:1.5rem;margin:0 0 4px}}h2{{font-size:1.05rem;margin:22px 0 8px}}h3{{font-size:.95rem;margin:16px 0 6px}}.muted{{color:var(--mut);font-size:.9rem}}
.m{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px}}.t{{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}}.d{{color:var(--mut);font-size:.85rem}}
.bar{{display:flex;height:24px;border-radius:6px;overflow:hidden;margin:8px 0 2px;font-size:.78rem;color:#fff}}.bar span{{display:flex;align-items:center;justify-content:center;min-width:0}}
.s0{{background:var(--a)}}.s1{{background:var(--b)}}.s2{{background:var(--c)}}.lg{{display:flex;justify-content:space-between;color:var(--mut);font-size:.75rem}}.lg i{{font-style:normal}}
.pk{{margin-top:6px;font-size:.9rem}}.mk{{color:var(--mut);font-size:.8rem}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--card);border:1px solid var(--line)}}th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}th{{color:var(--mut)}}
.ok{{color:#2e7d32;font-weight:700}}.no{{color:#c0392b;font-weight:700}}footer{{margin-top:26px;color:var(--mut);font-size:.8rem}}
</style></head><body>{radios}<main>
<h1>Football match predictions</h1>
<p class="muted">Machine-learning forecasts for the Premier League and Liga 1 Indonesia. Updated automatically: {updated}.</p>
<div class="tabs">{labels}</div>
{panels}
<footer>Data: football-data.co.uk (Premier League results and odds), Wikipedia and the league website (Liga 1). Open source: github.com/andriarezki/football-predictor. For learning only, not betting advice: no model here beats the bookmakers.</footer>
</main></body></html>"""


def epl_league() -> dict:
    paths = data.download(data.LIVE_SEASONS, refresh=("2627",))
    matches = data.parse(paths)
    fixtures = data.fetch_fixtures()
    pred = predict_fixtures(matches, fixtures) if not fixtures.empty else pd.DataFrame()
    tr = track_record(matches)
    return {"key": "epl", "name": "Premier League",
            "fixtures_html": render_fixtures(pred, "No Premier League fixtures are published yet. This fills in automatically when the next round appears (checked regularly)."),
            "track_html": render_track(tr, "{n} matches of the 2026/27 season, each predicted before it was played (the models only see earlier matches). The bookmaker line is the market's margin-free odds."),
            "note": "Model: logistic regression on Elo, recent form, goals and rest days."}


def liga1_league() -> dict:
    from . import liga1
    res = liga1.load_results()
    teams = sorted(set(res["2026-27"]["home"]) | set(res["2026-27"]["away"]))
    fx = liga1.fetch_fixtures(teams)
    pred = liga1.predict(res, fx) if not fx.empty else pd.DataFrame()
    tr = liga1.track_record(res)
    return {"key": "liga1", "name": "Liga 1 Indonesia",
            "fixtures_html": render_fixtures(pred, "No upcoming Liga 1 fixtures were found on the league website."),
            "track_html": render_track(tr, "Trained only on last season (2025/26), then scored on {n} matches of 2026/27 it had never seen. No bookmaker odds are available for Liga 1, so the benchmark is last season's home/draw/away split."),
            "note": "Model: time-decayed Poisson attack/defence model on Wikipedia results. Match dates are not available there, so the model only separates this season from last season. Venues can be neutral or borrowed, which weakens home advantage."}


def run(out_html: str, out_json: str | None = None) -> None:
    leagues = []
    for build in (epl_league, liga1_league):
        try:
            leagues.append(build())
        except Exception as e:  # one league failing must not blank the page
            name = "Premier League" if build is epl_league else "Liga 1 Indonesia"
            leagues.append({"key": "epl" if build is epl_league else "liga1", "name": name,
                            "fixtures_html": '<p class="muted">Data source unavailable right now. Will retry on the next run.</p>',
                            "track_html": "", "note": ""})
            print("league failed:", name, type(e).__name__, e)
    updated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(build_page(leagues, updated))
