"""
predictor.py — Single-match prediction engine (any national team, any time).

On top of the weighted Poisson strength model in strength.py, it adds:
  1) Dixon-Coles low-score correction for dependence among low-score outcomes
  2) A full match report derived from the scoreline probability matrix:
     win/draw/loss, expected goals and an interval, most likely scorelines,
     over/under, both-teams-to-score (BTTS) and clean-sheet probabilities.

The optional downloader uses a pinned public snapshot. Importing this module
does not access the network. Score grids include at least 1-1e-10 marginal mass.
"""
import os, sys, math, urllib.request
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
import pandas as pd
from strength import StrengthModel
from data_contract import ensure_data, load_results, SOURCE_URL

DATA = "results.csv"
DATA_URL = SOURCE_URL
MAXG = 12  # minimum score-grid size; the upper tail expands when required


def effective_rho(lh, la, rho):
    """Project rho into the match-specific region with all four tau > 0."""
    if not all(np.isfinite(v) for v in [lh, la, rho]) or min(lh, la) < 1e-12 or max(lh, la) > 100:
        raise ValueError("Expected goals must be in [1e-12, 100]; rho must be finite")
    epsilon = 1e-10
    lower = max(-1 / lh, -1 / la) + epsilon
    upper = min(1 / (lh * la), 1.0) - epsilon
    return float(np.clip(rho, lower, upper))


def _dc_tau(i, j, lh, la, rho):
    """Dixon-Coles low-score dependence-correction factor."""
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


class MatchPredictor:
    def __init__(self, cutoff=None, half_life=2.0, years_back=8):
        self.cutoff = cutoff or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.model = StrengthModel(half_life, years_back)
        self.rho = -0.05

    def fit(self, path=DATA):
        self.model.fit(path, cutoff=self.cutoff)
        self._fit_rho(path)
        return self

    def _fit_rho(self, path, sample_years=2):
        """Estimate the Dixon-Coles rho by MLE on recent matches."""
        df = load_results(path)
        cut = pd.Timestamp(self.cutoff)
        df = df[(df.date < cut) & (df.date >= cut - pd.DateOffset(years=sample_years))]
        df = df.dropna(subset=["home_score", "away_score"])
        # keep only matches where both teams exist in the fitted model
        ok = df.home_team.isin(self.model.teams) & df.away_team.isin(self.model.teams)
        df = df[ok]
        if len(df) > 1500:
            df = df.sample(1500, random_state=0)
        lams = []
        for h, a, hs, as_, n in zip(df.home_team, df.away_team,
                                    df.home_score, df.away_score, df.neutral):
            lh, la = self.model.expected_goals(h, a, neutral=bool(n))
            lams.append((lh, la, int(hs), int(as_)))

        if not lams:
            self.rho = 0.0
            return
        lh, la, hs, aws = np.asarray(lams).T
        lower = max(-.2, float(np.max(np.maximum(-1/lh, -1/la)))) + 1e-10
        upper = min(.2, float(np.min(np.minimum(1/(lh*la), 1.0)))) - 1e-10
        def negll(rho):
            # The independent-Poisson factors are constant in rho.
            tau = np.ones(len(lams))
            tau[(hs == 0) & (aws == 0)] = (1-lh*la*rho)[(hs == 0) & (aws == 0)]
            tau[(hs == 0) & (aws == 1)] = (1+lh*rho)[(hs == 0) & (aws == 1)]
            tau[(hs == 1) & (aws == 0)] = (1+la*rho)[(hs == 1) & (aws == 0)]
            tau[(hs == 1) & (aws == 1)] = 1-rho
            return float(-np.log(tau).sum()) if np.all(tau > 0) else np.inf
        res = minimize_scalar(negll, bounds=(lower, upper), method="bounded")
        if not res.success or not np.isfinite(res.fun):
            raise ValueError("Dixon-Coles optimization failed")
        self.rho = float(res.x)

    def score_matrix(self, home, away, neutral=True):
        lh, la = self.model.expected_goals(home, away, neutral=neutral)
        rho = effective_rho(lh, la, self.rho)
        if max(lh, la) > 100:
            raise ValueError("Expected goals exceed the supported numerical range (100)")
        size = max(MAXG, int(poisson.ppf(1-1e-10, max(lh, la))) + 1)
        ph = poisson.pmf(np.arange(size), lh)
        pa = poisson.pmf(np.arange(size), la)
        M = np.outer(ph, pa)
        for i in (0, 1):
            for j in (0, 1):
                M[i, j] *= _dc_tau(i, j, lh, la, rho)
        if not np.isfinite(M).all() or (M < 0).any() or M.sum() <= 0:
            raise ValueError("Invalid scoreline probability matrix")
        M /= M.sum()
        return M, lh, la

    def predict(self, home, away, neutral=True):
        M, lh, la = self.score_matrix(home, away, neutral)
        idx = np.indices(M.shape)
        home_w = M[idx[0] > idx[1]].sum()
        draw = M[idx[0] == idx[1]].sum()
        away_w = M[idx[0] < idx[1]].sum()
        total = idx[0] + idx[1]
        size = len(M)
        tot_dist = np.array([M[total == k].sum() for k in range(2 * size - 1)])
        cdf = np.cumsum(tot_dist)
        lo = int(np.searchsorted(cdf, 0.10)); hi = int(np.searchsorted(cdf, 0.90))
        top = sorted(((M[i, j], i, j) for i in range(size) for j in range(size)),
                     reverse=True)[:5]
        return dict(
            home=home, away=away, neutral=neutral,
            xg_home=lh, xg_away=la,
            p_home=home_w, p_draw=draw, p_away=away_w,
            exp_total=float((np.arange(len(tot_dist)) * tot_dist).sum()),
            total_mode=int(tot_dist.argmax()),
            total_lo=lo, total_hi=hi,
            top_scores=[(i, j, p) for p, i, j in top],
            over25=float(tot_dist[3:].sum()),
            btts=float(M[1:, 1:].sum()),
            cs_home=float(M[:, 0].sum()), cs_away=float(M[0, :].sum()),
            rho=self.rho,
            rho_effective=effective_rho(lh, la, self.rho),
            score_grid_size=size,
        )

    def report(self, home, away, neutral=True):
        r = self.predict(home, away, neutral)
        bar = lambda p: "█" * round(p * 30)
        site = "neutral venue" if neutral else f"{home} at home"
        print(f"\n{'='*54}\n  {home}  vs  {away}   ({site})\n{'='*54}")
        print(f"  Win / Draw / Loss")
        print(f"    {home:<16} {r['p_home']*100:5.1f}%  {bar(r['p_home'])}")
        print(f"    {'Draw':<16} {r['p_draw']*100:5.1f}%  {bar(r['p_draw'])}")
        print(f"    {away:<16} {r['p_away']*100:5.1f}%  {bar(r['p_away'])}")
        print(f"\n  Expected goals   {home} {r['xg_home']:.2f} - {r['xg_away']:.2f} {away}")
        print(f"  Total goals      most likely {r['total_mode']}, "
              f"80% interval {r['total_lo']}-{r['total_hi']}")
        print(f"\n  Most likely scorelines (top 5)")
        for i, j, p in r['top_scores']:
            print(f"    {home[:12]} {i}-{j} {away[:12]:<12}  {p*100:4.1f}%")
        print(f"\n  Over 2.5 goals  {r['over25']*100:4.1f}%    BTTS  {r['btts']*100:4.1f}%")
        print(f"  {home[:10]} clean sheet  {r['cs_home']*100:4.1f}%    "
              f"{away[:10]} clean sheet  {r['cs_away']*100:4.1f}%")
        print(f"{'='*54}")
        return r


if __name__ == "__main__":
    ensure_data(refresh=False)
    pred = MatchPredictor().fit()
    if len(sys.argv) >= 3:
        neutral = "--home" not in sys.argv
        pred.report(sys.argv[1], sys.argv[2], neutral=neutral)
    else:
        for h, a in [("France", "Spain"), ("Argentina", "Brazil"),
                     ("Morocco", "Portugal"), ("Japan", "Germany")]:
            pred.report(h, a)
