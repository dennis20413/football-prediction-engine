# Football Prediction Engine

[![checks](https://github.com/dennis20413/football-prediction-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/dennis20413/football-prediction-engine/actions/workflows/ci.yml)

Weighted Poisson match modelling, a Dixon-Coles score correction, and a reproducible chronological evaluation on international results. The repository also retains StatsBomb style/player analysis and an experimental tournament simulator.

**中文说明：**这个项目的重点是可复现的足球概率建模与评估。新版修正了基线准确率的计算，固定资料版本，并加入数据契约、概率有效性、时间隔离测试与 CI。比赛模拟采用简化规则；没有实时服务或真实交易表现验证。

## Verified evaluation

The model is fitted once on **2016-01-01 <= date < 2024-01-01** and evaluated on **2024-01-01 <= date < 2026-06-11**. Feature weights and correction parameters use training data exclusively. Hyperparameters are fixed; there is no selection using the test results.

| Metric | Match model | Same-training-window class-prior baseline |
|---|---:|---:|
| Log loss | 0.883128 | 1.053579 |
| Multiclass Brier, sum over three outcomes | 0.518313 | 0.635470 |
| Test accuracy | 58.8511% | 47.4404% |
| Total-goals MAE | 1.4000 | — |

There are 7,440 training matches and 2,564 candidate test matches. Five test matches contain a team absent from training and are excluded: **2,559 evaluated matches; coverage 99.805%**. This coverage restriction is included in the report.

The source is pinned to [international_results commit d5bd6de](https://github.com/martj42/international_results/tree/d5bd6de012c87e2f325de602ae77fe2f9ab61ef3); CSV SHA-256 is recorded in [reports/backtest.json](reports/backtest.json), together with package versions, intervals, parameters and reliability-bin counts. No raw dataset is committed.

**The old README's 49.1% baseline was a training class frequency, not measured baseline test accuracy.** The corrected table above supersedes that comparison. The pinned dataset also differs from the earlier mutable snapshot, so the old 2,542-match/59.0% numbers should not be combined with this new report. See [upgrade notes](docs/upgrade-notes.md).

Reliability bins are descriptive: for example, the 60–80% bin has mean prediction 68.90% and observed home-win rate 75.79% (570 matches). This is evidence of a gap, not a claim of perfect calibration. See all five bins in the report.

## Run and test

Python 3.11.6 was used for the committed report; offline CI covers Python 3.11 and 3.12 on Ubuntu and Windows.

~~~sh
python -m venv .venv
# Activate: source .venv/bin/activate (POSIX), or .venv\Scripts\Activate.ps1 (PowerShell)
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Explicit download of the pinned, checksum-checked public snapshot
python backtest.py --download --output reports/backtest.json

# A local CSV can instead be supplied; imports never download or train
python backtest.py --data results.csv --cutoff 2024-01-01 --test-end 2026-06-11
python predictor.py "France" "Spain"
~~~

A successful evaluation writes structured JSON and exits 0. Missing or invalid inputs, an empty evaluation interval, or an output path colliding with the data file exit 2. Download validation uses a temporary file; a failed refresh preserves the prior local snapshot. The sample CSV schema is date, home_team, away_team, home_score, away_score, tournament, neutral; extra provider columns are allowed.

The loader validates boolean venues, dates, nonnegative integer goals, paired missing scores and nonempty team identities. Identical model-relevant rows are deduplicated. Conflicting date/team/tournament identities inside selected training or test intervals are rejected. The pinned source contains two conflicting historical rows from 1974, outside both evaluation intervals; the report exposes this count.

## Model and probability guarantees

~~~mermaid
flowchart LR
  CSV[Pinned or local results CSV] --> Validate[Schema and identity validation]
  Validate --> Split[Fixed date intervals]
  Split --> Train[Training-only weighted Poisson]
  Train --> DC[Training-only Dixon-Coles parameter]
  DC --> Grid[Valid adaptive scoreline matrix]
  Grid --> WDL[Home / draw / away probabilities]
  WDL --> Report[Measured baseline, coverage and reliability]
~~~

The strength model uses time decay with a two-year half-life, fixed tournament importance weights and ridge regularization. Attack, defence and home advantage are fitted jointly. The correction parameter is estimated on a fixed recent training subset.

For each prediction, rho is projected into its admissible region so all four low-score correction factors remain positive. The effective and fitted values are exposed separately. No projections were required for the committed test predictions. The score grid expands until each independent Poisson marginal includes at least 1-1e-10 probability mass, then is normalized. Supported expected-goal rates are [1e-12, 100]; unsupported rates fail explicitly.

All match summaries—W/D/L, scorelines, goal interval, over 2.5, both teams scoring and clean sheets—come from the same matrix. Unknown or identical opposing teams fail clearly.

## Retained research modules

- **Style and player analysis:** analytics.py, players.py and matchup.py use [StatsBomb Open Data](https://github.com/statsbomb/open-data). Run their individual entry points after installation. They require network data, follow the provider's attribution/use requirements, and were not revalidated as part of this match-model upgrade. The stylistic matchup index is a heuristic.
- **Tournament scenarios:** run.py requires an explicit complete group-only fixture CSV: 12 disconnected groups of four, six unique pairings per group, 72 rows. Completed tournament results containing knockout matches are rejected rather than silently misidentified as groups.

~~~sh
python run.py --data results.csv --fixtures group_stage_fixtures.csv --simulations 1000 --seed 42
~~~

The simulator uses a simplified seeded knockout bracket, simplified tie breaks, independent Poisson goals and a strength-weighted penalty approximation. It is **not an implementation of every official tournament rule**. The former unsourced market-odds comparison has been removed. A fixed random seed and stable team ordering make supported scenarios repeatable.

## Tests and evidence

32 offline tests passed locally. Tests cover actual baseline test accuracy, date boundaries, final-test-score perturbation invariance, coverage/errors, extreme score probabilities, malformed CSVs, protected raw inputs, atomic refresh failure, shared calibration edges, tournament input guards and seed reproducibility. Synthetic tiny fixtures test code behavior; the result table above comes from the separately downloaded public dataset.

See [tests/test_evaluation.py](tests/test_evaluation.py), [reports/backtest.json](reports/backtest.json), [upgrade notes](docs/upgrade-notes.md), and [CI](https://github.com/dennis20413/football-prediction-engine/actions).

## Scope

This is a results-based research model with one fixed temporal split. Source scores can include extra time, so labels are not uniformly 90-minute outcomes. Team availability limits coverage, and reliability varies by probability bin. There is no claim of an accuracy ceiling, stable future performance, causal style advantages, live lineup ingestion or production deployment.
