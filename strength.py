"""
strength.py — Estimate each team's attack / defence strength from
historical international match results.

Method: Poisson regression (a simplified Dixon-Coles model)
    log(expected goals) = intercept + attack[scorer] - defence[conceder]
                          + home_advantage (if scorer is at home)

Each match is split into two rows (one per team's perspective) and fit with a
sparse one-hot design + PoissonRegressor, solving every team's attack/defence
coefficient at once. Two weighting schemes are applied: time decay (recent
matches weigh more) and match importance (competitive games > friendlies).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import PoissonRegressor

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 1.0, "FIFA World Cup qualification": 0.9,
    "UEFA Euro": 1.0, "UEFA Euro qualification": 0.8,
    "Copa América": 1.0, "African Cup of Nations": 0.9,
    "AFC Asian Cup": 0.9, "UEFA Nations League": 0.8,
    "Gold Cup": 0.7, "Confederations Cup": 0.8, "Friendly": 0.5,
}
DEFAULT_WEIGHT = 0.7


class StrengthModel:
    def __init__(self, half_life_years=2.0, years_back=8, ridge=1e-3):
        self.half_life = half_life_years
        self.years_back = years_back
        self.ridge = ridge

    def fit(self, path, cutoff):
        df = pd.read_csv(path, parse_dates=["date"])
        cut = pd.Timestamp(cutoff)
        df = df[(df.date < cut) & (df.date >= cut - pd.DateOffset(years=self.years_back))]
        df = df.dropna(subset=["home_score", "away_score"]).copy()

        # Time-decay weight (half-life in years) x match-importance weight
        age = (cut - df.date).dt.days / 365.25
        decay = 0.5 ** (age / self.half_life)
        importance = df.tournament.map(TOURNAMENT_WEIGHT).fillna(DEFAULT_WEIGHT)
        df["w"] = decay * importance

        # Long format: each match -> two rows (one per team)
        home = pd.DataFrame(dict(team=df.home_team, opp=df.away_team,
                                 goals=df.home_score.astype(int),
                                 home=np.where(df.neutral, 0, 1), w=df.w))
        away = pd.DataFrame(dict(team=df.away_team, opp=df.home_team,
                                 goals=df.away_score.astype(int),
                                 home=0, w=df.w))
        long = pd.concat([home, away], ignore_index=True)

        # Sparse one-hot design: attack columns (+1) and defence columns (-1)
        self.teams = sorted(set(long.team) | set(long.opp))
        idx = {t: i for i, t in enumerate(self.teams)}
        n, T = len(long), len(self.teams)
        rows = np.arange(n)
        att = sparse.csr_matrix((np.ones(n), (rows, long.team.map(idx))), shape=(n, T))
        deff = sparse.csr_matrix((-np.ones(n), (rows, long.opp.map(idx))), shape=(n, T))
        home_col = sparse.csr_matrix(long.home.values.reshape(-1, 1).astype(float))
        X = sparse.hstack([att, deff, home_col]).tocsr()

        self.model = PoissonRegressor(alpha=self.ridge, max_iter=500, fit_intercept=True)
        self.model.fit(X, long.goals.values, sample_weight=long.w.values)
        self._idx, self._T = idx, T
        return self

    def _vec(self, scorer, conceder, is_home):
        x = np.zeros(2 * self._T + 1)
        x[self._idx[scorer]] = 1
        x[self._T + self._idx[conceder]] = -1
        x[-1] = is_home
        return x.reshape(1, -1)

    def expected_goals(self, home_team, away_team, neutral=True):
        """Return (expected goals for home_team, expected goals for away_team)."""
        h = 0 if neutral else 1
        lam_h = float(self.model.predict(self._vec(home_team, away_team, h))[0])
        lam_a = float(self.model.predict(self._vec(away_team, home_team, 0))[0])
        return lam_h, lam_a


if __name__ == "__main__":
    m = StrengthModel().fit("results.csv", cutoff="2026-06-11")
    print(f"home advantage factor: {np.exp(m.model.coef_[-1]):.3f}x")
    for h, a in [("Spain", "Cape Verde"), ("France", "Norway"),
                 ("Brazil", "Morocco"), ("Germany", "Curaçao"),
                 ("Argentina", "Jordan")]:
        lh, la = m.expected_goals(h, a)
        print(f"{h:10s} vs {a:14s}: {lh:.2f} - {la:.2f}")
