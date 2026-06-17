"""
tournament.py — 2026 World Cup Monte Carlo simulation engine.

Key feature: it fixes the already-played real results and only simulates the
remaining fixtures. Format: 12 groups of 4 -> top 2 per group + 8 best
third-placed teams = 32 teams into the knockout stage.
"""

import numpy as np
import pandas as pd


def load_fixtures(path, season_start="2026-01-01"):
    df = pd.read_csv(path, parse_dates=["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= pd.Timestamp(season_start))]
    return wc.reset_index(drop=True)


def build_groups(fixtures):
    """Infer the 12 groups from the connectivity of the fixture list."""
    import networkx as nx
    G = nx.Graph()
    for _, r in fixtures.iterrows():
        G.add_edge(r.home_team, r.away_team)
    comps = sorted([sorted(c) for c in nx.connected_components(G)])
    return {chr(65 + i): set(g) for i, g in enumerate(comps)}


def sim_match(lam, a, b, rng):
    """Simulate one match using the precomputed expected-goals lookup.
    Returns (a goals, b goals, lambda_a, lambda_b)."""
    la, lb = lam[a][b], lam[b][a]
    return rng.poisson(la), rng.poisson(lb), la, lb


def _bracket_order(n):
    seeds = [1, 2]
    while len(seeds) < n:
        m = len(seeds) * 2 + 1
        seeds = [s for x in seeds for s in (x, m - x)]
    return seeds


class Tournament:
    def __init__(self, model, fixtures):
        self.model = model
        self.fixtures = fixtures
        self.groups = build_groups(fixtures)
        self.team_group = {t: g for g, ts in self.groups.items() for t in ts}
        # split played vs unplayed up front
        self.played = fixtures.dropna(subset=["home_score"]).copy()
        self.unplayed = fixtures[fixtures.home_score.isna()].copy()
        # precompute expected goals for every team pair; simulation just looks up
        teams = list(self.team_group)
        self.lam = {a: {} for a in teams}
        for a in teams:
            for b in teams:
                if a != b:
                    self.lam[a][b] = model.expected_goals(a, b)[0]

    def _group_table(self, results):
        """results: list of (home, away, hg, ag). Return ranked teams per group."""
        stats = {t: dict(pts=0, gf=0, ga=0) for t in self.team_group}
        for h, a, hg, ag in results:
            stats[h]["gf"] += hg; stats[h]["ga"] += ag
            stats[a]["gf"] += ag; stats[a]["ga"] += hg
            if hg > ag: stats[h]["pts"] += 3
            elif hg < ag: stats[a]["pts"] += 3
            else: stats[h]["pts"] += 1; stats[a]["pts"] += 1
        tables = {}
        for g, teams in self.groups.items():
            ranked = sorted(teams, key=lambda t: (
                stats[t]["pts"], stats[t]["gf"] - stats[t]["ga"], stats[t]["gf"],
                self._rng.random()), reverse=True)
            tables[g] = [(t, stats[t]) for t in ranked]
        return tables

    def _knockout(self, qualifiers):
        """qualifiers: 32 teams already ordered by seed. Single elimination."""
        order = _bracket_order(len(qualifiers))
        bracket = [qualifiers[s - 1] for s in order]  # place seeds into slots
        while len(bracket) > 1:
            nxt = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]
                ga, gb, la, lb = sim_match(self.lam, a, b, self._rng)
                if ga > gb: nxt.append(a)
                elif gb > ga: nxt.append(b)
                else:  # draw -> penalties, slightly weighted by strength
                    p = la / (la + lb)
                    nxt.append(a if self._rng.random() < p else b)
            bracket = nxt
        return bracket[0]

    def simulate_once(self, rng):
        self._rng = rng
        # 1) group stage: fix real results + simulate the unplayed fixtures
        results = [(r.home_team, r.away_team, int(r.home_score), int(r.away_score))
                   for _, r in self.played.iterrows()]
        for _, r in self.unplayed.iterrows():
            hg, ag, _, _ = sim_match(self.lam, r.home_team, r.away_team, rng)
            results.append((r.home_team, r.away_team, hg, ag))

        tables = self._group_table(results)
        # 2) qualification: top 2 per group + 8 best third-placed teams
        winners, runners, thirds = [], [], []
        for g in sorted(tables):
            ranked = tables[g]
            winners.append(ranked[0][0]); runners.append(ranked[1][0])
            t, s = ranked[2]
            thirds.append((t, s["pts"], s["gf"] - s["ga"], s["gf"]))
        thirds.sort(key=lambda x: (x[1], x[2], x[3], rng.random()), reverse=True)
        best_thirds = [t[0] for t in thirds[:8]]

        # 3) seeding: winners > runners-up > best thirds, then points / GD within
        def key(team):
            g = self.team_group[team]
            for rank, (t, s) in enumerate(tables[g]):
                if t == team:
                    return (rank, -s["pts"], -(s["gf"] - s["ga"]))
        qualifiers = sorted(winners + runners + best_thirds, key=key)
        return self._knockout(qualifiers)

    def monte_carlo(self, n_sims=20000, seed=42):
        rng = np.random.default_rng(seed)
        from collections import Counter
        champ = Counter()
        for _ in range(n_sims):
            champ[self.simulate_once(rng)] += 1
        out = pd.DataFrame(
            [(t, c / n_sims) for t, c in champ.items()],
            columns=["team", "win_prob"]).sort_values("win_prob", ascending=False)
        return out.reset_index(drop=True)
