# Football Prediction Engine ⚽ — Single-Match + Full-Tournament (with backtest)

Pre-match predictions for **any national team, at any time**. Data is
refreshed on every run, so it always reflects the latest results. Core: a
weighted Poisson strength model + Dixon-Coles low-score correction.

## What it tells you (single match)

```bash
python predictor.py "France" "Spain"
```
Outputs: win / draw / loss probabilities, expected goals for each side, the
most likely total-goals value with an 80% interval, the top 5 scorelines,
over/under 2.5, both-teams-to-score (BTTS), and clean-sheet probabilities.

## Accuracy (honest backtest, not a claim)

Trained on data before 2024, evaluated on **2,542 unseen matches** after 2024:

| Metric | Model | Baseline | Note |
|---|---|---|---|
| Log-loss | **0.884** | 1.054 | lower is better |
| Brier | **0.519** | 0.636 | lower is better |
| Accuracy | **59.0%** | 49.1% | predicting W/D/L |
| Goals MAE | 1.40 | — | total-goals error |

**Calibration is near-perfect:** matches the model rated 60-80% home win came
in at 76%; those rated 80%+ came in at 93%. The probabilities mean what they
say. Football is high-variance, so 59% accuracy is a solid result.
(Reproduce with `python backtest.py`.)

## Files

| File | Purpose |
|---|---|
| `strength.py` | Weighted Poisson strength model (sparse, memory-friendly) |
| `predictor.py` | **Single-match prediction engine** (Dixon-Coles + full report + CLI) |
| `backtest.py` | Out-of-time backtest, honest accuracy metrics |
| `tournament.py` | Full World Cup Monte Carlo (fixes played results, simulates the rest) |
| `run.py` | Title odds for the tournament (with market comparison for value) |

## Method

1. **Strength:** `log(expected goals) = intercept + attack[scorer] -
   defence[conceder] + home_advantage`, solving all teams at once; with time
   decay (2-year half-life) and match-importance weighting.
2. **Dixon-Coles correction:** rho estimated by MLE on recent matches, fixing
   the independent-Poisson bias on 0-0 / 1-0 / 1-1 (better accuracy and calibration).
3. **Scoreline matrix:** outer product of two Poisson distributions x the DC
   correction, then normalized. Every probability (W/D/L, over/under, BTTS,
   scorelines) is integrated from this single matrix.

## A note on "real-time" data (important)

- The data source, [martj42/international_results](https://github.com/martj42/international_results),
  **updates daily** and already includes in-progress World Cup results, which
  is current enough for pre-match prediction.
- True **in-play** data (lineups, injuries, live odds) requires a paid API
  (API-Football / Football-Data.org / Opta). The data layer is abstracted, so
  switching to a paid feed only means changing `ensure_data` and the loader.

## Possible extensions (portfolio value)

- Feed team-style interaction terms (press x build-up, counter x transition
  risk) into the model as features, and use the backtest to confirm which
  interactions are statistically significant (see the separate style analysis).
- Adjust strength for missing players (injuries / suspensions).
- Wrap it in Streamlit / FastAPI as a live query service for any fixture.
