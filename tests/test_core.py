import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analysis, data, evaluate, features  # noqa: E402
from src.poisson import PoissonModel  # noqa: E402


def fake_matches(n_rounds=40, seed=0):
    """Tiny league: 6 teams, 'A' is strong, 'F' is weak. Deterministic."""
    rng = np.random.default_rng(seed)
    teams = list("ABCDEF")
    strength = dict(zip(teams, [1.8, 1.4, 1.2, 1.0, 0.8, 0.5]))
    rows, d, mid = [], pd.Timestamp("2022-08-01"), 1
    for _ in range(n_rounds):
        for h in teams:
            a = teams[(teams.index(h) + 1 + rng.integers(0, 5)) % 6]
            hg, ag = rng.poisson(strength[h] * 1.2), rng.poisson(strength[a] * 0.9)
            rows.append({"match_id": mid, "season": "2223", "date": d, "home": h, "away": a, "hg": hg, "ag": ag,
                         "result": "H" if hg > ag else "D" if hg == ag else "A",
                         "odds_h": 2.0, "odds_d": 3.4, "odds_a": 3.8})
            mid += 1
            d += pd.Timedelta(days=2)
    return pd.DataFrame(rows)


def test_parse_real_format(tmp_path):
    csv = tmp_path / "E0_2425.csv"
    csv.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA\n"
                   "E0,16/08/2024,Man United,Fulham,1,0,H,1.5,4.5,6.0\n"
                   "E0,17/08/2024,Arsenal,Wolves,2,0,H,1.2,6.5,11\n")
    m = data.parse([csv])
    assert list(m["home"]) == ["Man United", "Arsenal"]
    assert m["season"].iloc[0] == "2425" and m["date"].is_monotonic_increasing


def test_features_do_not_leak():
    m = fake_matches()
    f = features.build(m)
    first = f.sort_values("date").iloc[0]
    assert pd.isna(first["form_h"]) and first["elo_diff"] == features.HOME_ELO  # nothing known yet
    # changing a match's score must not change its OWN features
    m2 = m.copy()
    m2.loc[50, ["hg", "ag"]] = [9, 0]
    f2 = features.build(m2)
    assert f.loc[f.match_id == m.loc[50, "match_id"], "elo_diff"].iloc[0] == f2.loc[f2.match_id == m.loc[50, "match_id"], "elo_diff"].iloc[0]


def test_poisson_probabilities_and_strength():
    m = fake_matches()
    pm = PoissonModel().fit(m, as_of=m["date"].max())
    p = pm.predict_proba("A", "F")
    assert abs(sum(p) - 1) < 1e-9
    assert p[0] > p[2]  # strong home side favoured
    assert pm.predict_proba("A", "NEW") == (0.45, 0.25, 0.30)


def test_metrics():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert evaluate.brier(perfect, y) == 0
    assert evaluate.log_loss(perfect, y) < 1e-6
    uniform = np.full((3, 3), 1 / 3)
    assert abs(evaluate.log_loss(uniform, y) - np.log(3)) < 1e-9


def test_market_probs_sum_to_one():
    df = pd.DataFrame({"odds_h": [2.0], "odds_d": [3.4], "odds_a": [3.8]})
    assert abs(evaluate.market_probs(df).sum() - 1) < 1e-9


def test_backtest_runs_and_beats_nothing_by_accident():
    m = fake_matches(n_rounds=70)
    f = features.build(m)
    bt = evaluate.backtest(f, test_from="2023-02-01")
    s = evaluate.summarise(bt).set_index("model")
    assert {"poisson", "logistic", "grad_boost", "baseline_freq", "bookmaker"} <= set(s.index)
    assert s.loc["poisson", "log_loss"] < s.loc["baseline_freq", "log_loss"]  # strength is learnable here


def test_sql_queries(tmp_path):
    m = fake_matches()
    db = tmp_path / "t.db"
    data.to_sqlite(m, db)
    with sqlite3.connect(db) as con:
        q = analysis.run_queries(con)
    assert set(q) == {"home_advantage", "final_table", "rolling_form", "biggest_overperformers"}
    table = q["final_table"]
    assert table["pos"].min() == 1
    assert table["played"].sum() == 2 * len(m)


def test_live_prediction_and_page():
    from src import live

    m = fake_matches(n_rounds=60)
    fx = pd.DataFrame([{"date": pd.Timestamp("2023-06-01"), "Time": "15:00", "home": "A", "away": "F",
                        "odds_h": 1.5, "odds_d": 4.0, "odds_a": 7.0}])
    p = live.predict_fixtures(m, fx)
    assert abs(sum(p.loc[0, "p"]) - 1) < 1e-9 and p.loc[0, "p"][0] > p.loc[0, "p"][2]
    league = {"key": "epl", "name": "Premier League", "fixtures_html": live.render_fixtures(p, "none"),
              "track_html": live.render_track({"n": 0}, ""), "note": ""}
    empty = {"key": "liga1", "name": "Liga 1 Indonesia", "fixtures_html": live.render_fixtures(pd.DataFrame(), "No fixtures"),
             "track_html": "", "note": ""}
    page = live.build_page([league, empty], "test")
    assert "A</b> vs <b>F" in page and "not betting advice" in page
    assert "No fixtures" in page and "Liga 1 Indonesia" in page


def test_liga1_parsing():
    from src import liga1

    raw = """|name_AFC={{nowrap|[[Arema F.C.|Arema]]}}
|name_PSB={{nowrap|[[Persib Bandung|Persib]]}}
|match_AFC_PSB=1–3
|match_PSB_AFC=
"""
    names, df = liga1.parse_wikitext(raw)
    assert names == {"AFC": "Arema", "PSB": "Persib"} and df.to_dict("records") == [{"home": "Arema", "away": "Persib", "hg": 1, "ag": 3}]
    html_ = """<div align="left" style="margin-top:10px" class="col-md-12">
        Friday, 06 November 2026</div><table><tr height="25" class="table-row"><td><b>AREMA FC</b></td>
        <td><span>19:00</span></td><td><b>PERSIB BANDUNG</b></td><td width="30%" align="left">
        Some Stadium </td></tr></table>"""
    fx = liga1.parse_fixtures(html_, ["Arema", "Persib"])
    assert fx.loc[0, "home"] == "Arema" and fx.loc[0, "away"] == "Persib" and fx.loc[0, "time"] == "19:00"
