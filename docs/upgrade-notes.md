# Evaluation and engineering upgrade — 2026-10-28

The previous implementation had a genuine pre-cutoff model, but printed the training majority frequency as baseline accuracy. The corrected evaluator computes the baseline's predictions on the same test labels as the model and uses the same eight-year source window. It exposes coverage after excluding unseen teams.

The committed report uses a pinned public CSV, SHA-256 `5dcefd1aa0483979ea8378cef5be448d2005609d39945ea162378561a30a6767`, retrieved from source commit `d5bd6de012c87e2f325de602ae77fe2f9ab61ef3`. It evaluates 2,559 of 2,564 candidate matches. Model accuracy is 58.8511%; baseline accuracy is 47.4404%; log losses are 0.883128 and 1.053579. The older mutable source produced a different sample. Replace the old sample count and baseline together when updating a resume.

Model parameters were not selected using these test metrics. The upgrade adds validated inputs, training-only parameter fitting tests, an admissible Dixon-Coles correction and adaptive probability tails. The vectorized rho objective omits only the independent-Poisson factors that are constant with respect to rho. The fitted rho is 0.018779; no held-out prediction required per-match projection.

Importing `backtest` no longer downloads data or starts model training. Downloads are explicit, pinned and checksum checked; failed downloads preserve prior local data. Outputs cannot overwrite the input path or an existing hard link to it. JSON contains the source fingerprint, runtime versions, interval counts, model parameters, measured baseline and reliability bins.

Calibration wording was narrowed to match measured evidence. The README no longer calls calibration near-perfect or claims a universal accuracy ceiling. Source scores can include extra time, which limits interpretation of outcome probabilities.

The tournament entry point now requires group-only fixtures and rejects knockout-connected inputs. Its approximate bracket and tie breaks are documented; unsupported market-value comparisons were removed. The retained StatsBomb analysis modules were not independently revalidated in this upgrade.

Validation: 32 offline tests passed on Windows with Python 3.11.6. The full public-data backtest also completed successfully. The environment emits a non-fatal warning when the optional physical-core probe is unavailable; computation falls back to logical-core detection. CI is configured for Ubuntu/Windows and Python 3.11/3.12. Remote CI results are available in Actions after publication.

Before changes, a complete Git bundle was saved locally as `outputs/portfolio-backups/football-prediction-engine-before-upgrade.bundle`. That backup is not part of the public repository.
