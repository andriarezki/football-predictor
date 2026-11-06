"""Build a self-contained HTML dashboard (charts embedded as base64 PNG)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COL = {"bookmaker": "#c0392b", "logistic": "#2c7a7b", "poisson": "#6b5b95", "grad_boost": "#d98e04", "baseline_freq": "#9aa5b1"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_models(summary: pd.DataFrame) -> str:
    s = summary.sort_values("log_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.barh(s["model"], s["log_loss"], color=[COL.get(m, "#444") for m in s["model"]])
    for i, v in enumerate(s["log_loss"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.9, s["log_loss"].max() + 0.05)
    ax.set_xlabel("log loss on unseen matches (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def chart_calibration(bt: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=1)
    for name in ("bookmaker", "logistic"):
        g = bt[bt["model"] == name]
        bins = pd.cut(g["pH"], np.linspace(0, 1, 9))
        d = g.assign(hit=(g["y"] == 0).astype(int)).groupby(bins, observed=True).agg(p=("pH", "mean"), obs=("hit", "mean"), n=("hit", "size"))
        d = d[d["n"] >= 15]
        ax.plot(d["p"], d["obs"], "o-", color=COL[name], label=name)
    ax.set_xlabel("predicted home-win probability")
    ax.set_ylabel("share of home wins that happened")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def chart_cumulative(bt: pd.DataFrame) -> str:
    book = bt[bt["model"] == "bookmaker"].set_index("match_id")
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for name in ("logistic", "poisson", "grad_boost"):
        g = bt[bt["model"] == name].set_index("match_id").loc[book.index]
        ll = -np.log(np.clip(g[["pH", "pD", "pA"]].to_numpy()[np.arange(len(g)), g["y"].astype(int)], 1e-9, 1))
        lb = -np.log(np.clip(book[["pH", "pD", "pA"]].to_numpy()[np.arange(len(book)), book["y"].astype(int)], 1e-9, 1))
        ax.plot(book["date"].to_numpy(), np.cumsum(ll - lb), color=COL[name], label=name)
    ax.axhline(0, color="#bbb", lw=1)
    ax.set_ylabel("extra log loss vs bookmaker (cumulative)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    return _png(fig)


def chart_home(home: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(home))
    ax.bar(x, home["home_win_pct"], color="#2c7a7b", label="home win %")
    ax.bar(x, home["draw_pct"], bottom=home["home_win_pct"], color="#9aa5b1", label="draw %")
    ax.bar(x, home["away_win_pct"], bottom=home["home_win_pct"] + home["draw_pct"], color="#d98e04", label="away win %")
    ax.set_xticks(x, [f"20{s[:2]}/{s[2:]}" for s in home["season"]])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def build(summary: pd.DataFrame, bt: pd.DataFrame, queries: dict[str, pd.DataFrame], out: Path) -> None:
    rows = "".join(
        f"<tr><td>{r.model}</td><td>{r.matches}</td><td>{r.accuracy:.1%}</td><td>{r.brier:.4f}</td><td>{r.log_loss:.4f}</td></tr>"
        for r in summary.itertuples()
    )
    over = queries["biggest_overperformers"].head(5)
    over_rows = "".join(
        f"<tr><td>20{r.season[:2]}/{r.season[2:]}</td><td>{r.team}</td><td>{int(r.points)}</td><td>{r.market_expected_points}</td><td>{r.diff:+.1f}</td></tr>"
        for r in over.itertuples()
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football match predictor: results</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#f7f8f6;color:#1d2733}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:1.6rem;margin:0 0 4px}} h2{{font-size:1.05rem;margin:28px 0 8px}}
.sub{{color:#52606d;margin:0 0 16px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #e1e5e9;border-radius:10px;padding:14px}} img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #eceff1}}
th{{color:#52606d;font-weight:600}} .note{{font-size:.88rem;color:#52606d}}
</style></head><body><main>
<h1>Football match predictor</h1>
<p class="sub">Premier League 2019/20 to 2024/25 &middot; walk-forward backtest on {int(summary['matches'].max())} unseen matches (Aug 2023 to May 2025)</p>
<div class="grid">
<div class="card"><h2>Model vs bookmaker</h2><table><tr><th>model</th><th>matches</th><th>accuracy</th><th>Brier</th><th>log loss</th></tr>{rows}</table>
<p class="note">Lower Brier and log loss is better. The bookmaker line is the average market odds with the margin removed.</p></div>
<div class="card"><h2>Log loss by model</h2><img alt="log loss by model" src="data:image/png;base64,{chart_models(summary)}"></div>
<div class="card"><h2>How far behind the market?</h2><img alt="cumulative log loss vs bookmaker" src="data:image/png;base64,{chart_cumulative(bt)}"></div>
<div class="card"><h2>Calibration (home win)</h2><img alt="calibration" src="data:image/png;base64,{chart_calibration(bt)}"></div>
<div class="card"><h2>Results mix by season</h2><img alt="home advantage by season" src="data:image/png;base64,{chart_home(queries['home_advantage'])}"></div>
<div class="card"><h2>Season points vs market expectation</h2><table><tr><th>season</th><th>team</th><th>points</th><th>market xPts</th><th>diff</th></tr>{over_rows}</table>
<p class="note">Computed in SQL from average odds. Big positive numbers are teams that beat what the market expected.</p></div>
</div>
<p class="note">Data: football-data.co.uk (results and average odds). Not betting advice: no model here beats the market.</p>
</main></body></html>"""
    out.write_text(html, encoding="utf-8")
