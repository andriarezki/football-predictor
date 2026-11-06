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


def build_page(fixtures: pd.DataFrame, tr: dict, updated: str) -> str:
    def bar(p, mk=None):
        seg = "".join(f'<span class="s{i}" style="width:{v*100:.1f}%">{v*100:.0f}%</span>' for i, v in enumerate(p))
        return f'<div class="bar">{seg}</div>'
    if fixtures.empty:
        fx_html = '<p class="muted">Belum ada jadwal Premier League yang terbit di sumber data. Halaman ini otomatis terisi begitu jadwal pekan berikutnya muncul (dicek Selasa dan Jumat).</p>'
    else:
        fx_html = ""
        for r in fixtures.itertuples():
            pick = ["Menang kandang", "Seri", "Menang tandang"][int(np.argmax(r.p))]
            mk = f'<div class="mk">Pasar: {" / ".join(f"{v*100:.0f}%" for v in r.market)}</div>' if r.market else ""
            fx_html += (f'<div class="m"><div class="t"><b>{html.escape(r.home)}</b> vs <b>{html.escape(r.away)}</b>'
                        f'<span class="d">{r.date} {r.time}</span></div>{bar(r.p)}<div class="lg"><i>Kandang</i><i>Seri</i><i>Tandang</i></div>'
                        f'<div class="pk">Condong: {pick}</div>{mk}</div>')
    if tr.get("n"):
        m = tr["models"]
        rows = "".join(f"<tr><td>{k}</td><td>{v['accuracy']*100:.1f}%</td><td>{v['brier']:.3f}</td><td>{v['log_loss']:.3f}</td></tr>"
                       for k, v in m.items())
        rec = ""
        for r in tr["recent"]:
            hit = "✓" if int(np.argmax(r["p"])) == r["y"] else "✗"
            rec += (f'<tr><td>{r["date"][5:]}</td><td>{html.escape(r["home"])} {r["score"]} {html.escape(r["away"])}</td>'
                    f'<td>{max(r["p"])*100:.0f}%</td><td class="{"ok" if hit=="✓" else "no"}">{hit}</td></tr>')
        tr_html = (f'<p class="muted">{tr["n"]} laga musim 2026/27 yang diprediksi sebelum dimainkan (model hanya melihat laga sebelumnya).</p>'
                   f'<table><tr><th>model</th><th>akurasi</th><th>Brier</th><th>log loss</th></tr>{rows}</table>'
                   f'<h3>10 laga terakhir</h3><table><tr><th>tgl</th><th>hasil</th><th>keyakinan</th><th></th></tr>{rec}</table>')
    else:
        tr_html = '<p class="muted">Belum cukup laga musim ini.</p>'
    return f"""<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Prediksi Premier League</title>
<style>
:root{{--bg:#f7f8f6;--fg:#1d2733;--card:#fff;--line:#e1e5e9;--mut:#52606d;--a:#2c7a7b;--b:#9aa5b1;--c:#d98e04}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181c;--fg:#e8ecef;--card:#1d2328;--line:#2d353c;--mut:#9aa7b2}}}}
body{{font-family:system-ui,Arial,sans-serif;margin:0;background:var(--bg);color:var(--fg)}}main{{max-width:760px;margin:0 auto;padding:20px 16px 48px}}
h1{{font-size:1.5rem;margin:0 0 4px}}h2{{font-size:1.05rem;margin:26px 0 8px}}h3{{font-size:.95rem;margin:16px 0 6px}}.muted{{color:var(--mut);font-size:.9rem}}
.m{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px}}.t{{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}}.d{{color:var(--mut);font-size:.85rem}}
.bar{{display:flex;height:24px;border-radius:6px;overflow:hidden;margin:8px 0 2px;font-size:.78rem;color:#fff}}.bar span{{display:flex;align-items:center;justify-content:center;min-width:0}}
.s0{{background:var(--a)}}.s1{{background:var(--b)}}.s2{{background:var(--c)}}.lg{{display:flex;justify-content:space-between;color:var(--mut);font-size:.75rem}}.lg i{{font-style:normal}}
.pk{{margin-top:6px;font-size:.9rem}}.mk{{color:var(--mut);font-size:.8rem}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;background:var(--card);border:1px solid var(--line);border-radius:8px}}th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}th{{color:var(--mut)}}
.ok{{color:#2e7d32;font-weight:700}}.no{{color:#c0392b;font-weight:700}}footer{{margin-top:26px;color:var(--mut);font-size:.8rem}}
</style></head><body><main>
<h1>Prediksi Premier League</h1><p class="muted">Model machine learning (regresi logistik: Elo, form, gol, hari istirahat). Diperbarui otomatis: {updated}.</p>
<h2>Pertandingan berikutnya</h2>{fx_html}
<h2>Rapor model musim ini</h2>{tr_html}
<footer>Data: football-data.co.uk. Proyek terbuka: github.com/andriarezki/football-predictor. Hanya untuk belajar, bukan saran taruhan; model belum mengalahkan pasar.</footer>
</main></body></html>"""


def run(out_html: str, out_json: str | None = None) -> None:
    paths = data.download(data.LIVE_SEASONS, refresh=("2627",))
    matches = data.parse(paths)
    fixtures = data.fetch_fixtures()
    pred = predict_fixtures(matches, fixtures) if not fixtures.empty else pd.DataFrame()
    tr = track_record(matches)
    updated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(build_page(pred, tr, updated))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"updated": updated, "fixtures": pred.to_dict("records"), "track": tr}, f, default=str)
