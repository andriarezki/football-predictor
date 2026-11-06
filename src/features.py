"""Leak-free pre-match features: Elo, rolling form, goal averages, rest days."""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

K, HOME_ELO = 20.0, 60.0
FEATURES = ["elo_diff", "form_h", "form_a", "gf_h", "gf_a", "ga_h", "ga_a", "rest_h", "rest_a"]


def build(matches: pd.DataFrame) -> pd.DataFrame:
    """One pass through matches in date order. Features use only earlier matches."""
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=5))  # (points, gf, ga)
    last = {}
    rows = []
    for m in matches.sort_values(["date", "match_id"]).itertuples():
        h, a = m.home, m.away
        def avg(t, i):
            d = hist[t]
            return sum(x[i] for x in d) / len(d) if d else None
        rows.append({
            "match_id": m.match_id,
            "elo_diff": elo[h] + HOME_ELO - elo[a],
            "form_h": avg(h, 0), "form_a": avg(a, 0),
            "gf_h": avg(h, 1), "gf_a": avg(a, 1), "ga_h": avg(h, 2), "ga_a": avg(a, 2),
            "rest_h": min((m.date - last[h]).days, 14) if h in last else 7,
            "rest_a": min((m.date - last[a]).days, 14) if a in last else 7,
        })
        exp_h = 1 / (1 + 10 ** (-(elo[h] + HOME_ELO - elo[a]) / 400))
        s_h = 1.0 if m.hg > m.ag else 0.5 if m.hg == m.ag else 0.0
        gd = abs(m.hg - m.ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = K * mult * (s_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
        hist[h].append((3 if m.hg > m.ag else 1 if m.hg == m.ag else 0, m.hg, m.ag))
        hist[a].append((3 if m.ag > m.hg else 1 if m.ag == m.hg else 0, m.ag, m.hg))
        last[h] = last[a] = m.date
    f = pd.DataFrame(rows)
    out = matches.merge(f, on="match_id")
    return out
