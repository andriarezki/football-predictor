"""Attack/defence Poisson model with time decay (a simplified Dixon-Coles idea)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

MAX_GOALS = 10


class PoissonModel:
    def __init__(self, half_life_days: float = 180, alpha: float = 0.01):
        self.xi = np.log(2) / half_life_days
        self.alpha = alpha

    def _design(self, home, away, teams):
        idx = {t: i for i, t in enumerate(teams)}
        n, k = len(home), len(teams)
        X = np.zeros((2 * n, 2 * k + 1))
        for r, (h, a) in enumerate(zip(home, away)):
            X[r, idx[h]] = 1            # home attack
            X[r, k + idx[a]] = 1        # away defence
            X[n + r, idx[a]] = 1        # away attack
            X[n + r, k + idx[h]] = 1    # home defence
            X[r, 2 * k] = 1             # home advantage on home goals only
        return X

    def fit(self, df: pd.DataFrame, as_of: pd.Timestamp):
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        X = self._design(df["home"].tolist(), df["away"].tolist(), self.teams)
        y = np.concatenate([df["hg"].to_numpy(), df["ag"].to_numpy()])
        age = (as_of - df["date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * age)
        self.m = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(X, y, sample_weight=np.concatenate([w, w]))
        return self

    def predict_proba(self, home: str, away: str) -> tuple[float, float, float]:
        """Return (P home win, P draw, P away win). Unknown teams get a league-average guess."""
        if home not in self.teams or away not in self.teams:
            return 0.45, 0.25, 0.30
        X = self._design([home], [away], self.teams)
        lh, la = self.m.predict(X)
        g = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        grid /= grid.sum()
        return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())
