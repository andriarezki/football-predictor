"""Walk-forward backtest: retrain each month on the past only, predict that month."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES
from .poisson import PoissonModel

CLASSES = ["H", "D", "A"]


def market_probs(df: pd.DataFrame) -> np.ndarray:
    inv = 1 / df[["odds_h", "odds_d", "odds_a"]].to_numpy()
    return inv / inv.sum(axis=1, keepdims=True)  # remove the bookmaker margin


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))


def backtest(df: pd.DataFrame, test_from: str = "2023-08-01", poisson_every_n_months: int = 1) -> pd.DataFrame:
    """Return one row per test match with every model's (pH, pD, pA)."""
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    df["y"] = df["result"].map({c: i for i, c in enumerate(CLASSES)})
    start = pd.Timestamp(test_from)
    months = pd.period_range(start, df["date"].max(), freq="M")
    out = []
    for per in months:
        test = df[(df["date"].dt.to_period("M") == per)]
        train = df[df["date"] < per.to_timestamp()]
        if test.empty or len(train) < 300:
            continue
        tr_f = train.dropna(subset=FEATURES)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(tr_f[FEATURES], tr_f["y"])
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, random_state=0).fit(tr_f[FEATURES], tr_f["y"])
        pm = PoissonModel().fit(train, as_of=per.to_timestamp())
        te = test.copy()
        Xf = te[FEATURES].fillna(te[FEATURES].median())
        P_lr, P_gb = lr.predict_proba(Xf), gb.predict_proba(Xf)
        P_po = np.array([pm.predict_proba(h, a) for h, a in zip(te["home"], te["away"])])
        freq = train["y"].value_counts(normalize=True).reindex([0, 1, 2]).to_numpy()
        for name, P in [("baseline_freq", np.tile(freq, (len(te), 1))), ("poisson", P_po),
                        ("logistic", P_lr), ("grad_boost", P_gb)]:
            for i, (idx, row) in enumerate(te.iterrows()):
                out.append({"match_id": row.match_id, "date": row.date, "model": name, "y": row.y,
                            "pH": P[i, 0], "pD": P[i, 1], "pA": P[i, 2]})
        if te["odds_h"].notna().all():
            Pm = market_probs(te)
            for i, (idx, row) in enumerate(te.iterrows()):
                out.append({"match_id": row.match_id, "date": row.date, "model": "bookmaker", "y": row.y,
                            "pH": Pm[i, 0], "pD": Pm[i, 1], "pA": Pm[i, 2]})
    return pd.DataFrame(out)


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in bt.groupby("model"):
        P, y = g[["pH", "pD", "pA"]].to_numpy(), g["y"].to_numpy().astype(int)
        rows.append({"model": name, "matches": len(g), "accuracy": round(float((P.argmax(1) == y).mean()), 4),
                     "brier": round(brier(P, y), 4), "log_loss": round(log_loss(P, y), 4)})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
