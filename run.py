"""
run.py — 2026 World Cup title odds: one-command entry point.

Pipeline: download data -> estimate strength (Poisson) -> Monte Carlo the
whole tournament -> compare against market-implied probabilities.
"""
import os, time, urllib.request
import pandas as pd
from strength import StrengthModel
from tournament import load_fixtures, Tournament

DATA = "results.csv"
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CUTOFF = "2026-06-11"   # tournament start: only use matches before this to fit strength
N_SIMS = 10000

# Market-implied probabilities (reference only, as of 2026-06-15; update later).
# Source: Polymarket / bookmaker odds, de-vigged approximation.
MARKET = {
    "France": 0.16, "Spain": 0.16, "England": 0.11, "Brazil": 0.09,
    "Argentina": 0.09, "Portugal": 0.07, "Germany": 0.05,
}


def ensure_data():
    if not os.path.exists(DATA):
        print("Downloading historical international results...")
        urllib.request.urlretrieve(DATA_URL, DATA)


def main():
    ensure_data()
    print("1/3  Estimating team strength (weighted Poisson regression)...")
    model = StrengthModel(half_life_years=2.0, years_back=8).fit(DATA, cutoff=CUTOFF)

    print("2/3  Loading fixtures (including played results)...")
    tour = Tournament(model, load_fixtures(DATA))
    print(f"     {len(tour.played)} played, {len(tour.unplayed)} to simulate")

    print(f"3/3  Monte Carlo: {N_SIMS} simulations...")
    t0 = time.time()
    probs = tour.monte_carlo(n_sims=N_SIMS)
    print(f"     done in {time.time()-t0:.0f}s\n")

    probs.to_csv("win_probabilities.csv", index=False)
    mkt = dict(MARKET)
    print("=== 2026 World Cup title odds (Top 15) ===")
    print(f"{'#':>2}  {'Team':<16}{'Model':>8}{'Market':>8}{'Edge':>8}")
    for i, r in probs.head(15).iterrows():
        m = mkt.get(r.team)
        ms = f"{m*100:6.1f}%" if m else "    -- "
        diff = f"{(r.win_prob-m)*100:+6.1f}" if m else "    --"
        print(f"{i+1:>2}. {r.team:<16}{r.win_prob*100:6.1f}%{ms}{diff}")
    print("\nModel > Market = model thinks the team is undervalued (possible value); vice versa.")
    print("Wrote win_probabilities.csv")


if __name__ == "__main__":
    main()
