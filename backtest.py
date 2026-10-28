"""Frozen chronological evaluation; importing this module has no side effects."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from data_contract import ensure_data, load_results, sha256, assert_unique_matches, SOURCE_COMMIT, SOURCE_URL
from predictor import DATA, MatchPredictor

TRAIN_CUT = "2024-01-01"
TEST_END = "2026-06-11"


def outcome(home, away):
    return np.where(np.asarray(home) > np.asarray(away), 0,
                    np.where(np.asarray(home) == np.asarray(away), 1, 2)).astype(int)


def metrics(probabilities, labels):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(labels)
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p) or len(p) == 0:
        raise ValueError("Expected nonempty n-by-3 probabilities and n labels")
    if y.ndim != 1 or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Outcome labels must be H=0, D=1, A=2")
    y = y.astype(int)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Probabilities must be finite, nonnegative and sum to one")
    return {"log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            "brier_multiclass_sum": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
            "accuracy": float(np.mean(p.argmax(axis=1) == y))}


def reliability(probabilities, labels):
    edges = np.linspace(0, 1, 6)
    p = np.asarray(probabilities)
    home = np.asarray(labels) == 0
    indices = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
    bins = []
    for i, (low, high) in enumerate(zip(edges, edges[1:])):
        mask = indices == i
        bins.append({"lower": float(low), "upper": float(high), "matches": int(mask.sum()),
                     "mean_predicted_home_win": float(p[mask].mean()) if mask.any() else None,
                     "observed_home_win": float(home[mask].mean()) if mask.any() else None})
    assert sum(row["matches"] for row in bins) == len(p)
    return bins


def split_matches(frame, cutoff, test_end, years_back=8):
    cut, end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    if pd.isna(cut) or pd.isna(end) or end <= cut:
        raise ValueError("test_end must be after cutoff")
    played = frame.dropna(subset=["home_score", "away_score"])
    train = played[(played.date < cut) & (played.date >= cut - pd.DateOffset(years=years_back))]
    test = played[(played.date >= cut) & (played.date < end)]
    if train.empty or test.empty:
        raise ValueError("Training and test intervals both need played matches")
    assert_unique_matches(train)
    assert_unique_matches(test)
    return train.copy(), test.copy()


def evaluate(path, cutoff=TRAIN_CUT, test_end=TEST_END, half_life=2.0, years_back=8):
    before_hash = sha256(path)
    frame = load_results(path)
    train, test = split_matches(frame, cutoff, test_end, years_back)
    pred = MatchPredictor(cutoff=cutoff, half_life=half_life, years_back=years_back).fit(path)
    eligible = test.home_team.isin(pred.model.teams) & test.away_team.isin(pred.model.teams)
    excluded = int((~eligible).sum())
    test = test.loc[eligible]
    if test.empty:
        raise ValueError("No test matches have both teams represented in training")
    labels = outcome(test.home_score, test.away_score)
    predictions = [pred.predict(row.home_team, row.away_team, neutral=bool(row.neutral))
                   for row in test.itertuples()]
    p = np.asarray([[row["p_home"], row["p_draw"], row["p_away"]] for row in predictions])
    counts = np.bincount(outcome(train.home_score, train.away_score), minlength=3)
    prior = counts / counts.sum()
    baseline = np.tile(prior, (len(test), 1))
    model_metrics = metrics(p, labels)
    model_metrics["total_goals_mae"] = float(np.mean(np.abs(
        np.asarray([row["exp_total"] for row in predictions]) - (test.home_score + test.away_score).to_numpy())))
    if sha256(path) != before_hash:
        raise ValueError("Input changed during evaluation")
    return {"protocol": "Frozen pre-cutoff model; no test-time refits or hyperparameter selection",
            "source": {"file": Path(path).name, "sha256": before_hash, "rows": len(frame),
                       "pinned_download_commit": SOURCE_COMMIT, "pinned_download_url": SOURCE_URL,
                       "exact_duplicates_removed": frame.attrs.get("exact_duplicates_removed", 0),
                       "conflicting_duplicate_rows_full_history": frame.attrs.get("conflicting_duplicate_rows", 0)},
            "intervals": {"training_start_inclusive": (pd.Timestamp(cutoff) - pd.DateOffset(years=years_back)).strftime("%Y-%m-%d"),
                          "training_end_exclusive": cutoff, "test_start_inclusive": cutoff,
                          "test_end_exclusive": test_end, "training_matches": len(train),
                          "candidate_test_matches": len(test) + excluded, "evaluated_test_matches": len(test),
                          "excluded_unseen_team_matches": excluded,
                          "coverage": float(len(test) / (len(test) + excluded))},
            "parameters": {"half_life_years": half_life, "years_back": years_back, "ridge": .001,
                           "rho": pred.rho, "rho_projected_test_matches": sum(abs(row["rho"] - row["rho_effective"]) > 1e-12 for row in predictions)},
            "baseline_training_prior_hda": prior.tolist(),
            "metrics": {"model": model_metrics, "same_window_prior": metrics(baseline, labels)},
            "home_win_reliability": reliability(p[:, 0], labels),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "limitations": ["Single fixed temporal split; no claim of cross-competition generalization",
                            "Baseline accuracy is measured on test labels, not training class frequency",
                            "Score source includes extra time where recorded; outcomes are not strictly regulation-time labels",
                            "Rho is projected to valid bounds for each prediction; adaptive score tail is truncated at 1e-10 marginal mass",
                            "Calibration bins are descriptive and provide no proof of near-perfect calibration",
                            "No live feature feeds, production deployment, or wagering-performance evaluation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(DATA))
    parser.add_argument("--download", action="store_true", help="Download the pinned public snapshot")
    parser.add_argument("--cutoff", default=TRAIN_CUT)
    parser.add_argument("--test-end", default=TEST_END)
    parser.add_argument("--output", type=Path, default=Path("reports/backtest.json"))
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.data.resolve() or (
                args.output.exists() and args.data.exists() and args.output.samefile(args.data)):
            raise ValueError("Evaluation output must not overwrite source data")
        if args.download:
            ensure_data(args.data, refresh=True)
        if not args.data.exists():
            raise ValueError("No local data; pass --download or --data PATH")
        report = evaluate(args.data, args.cutoff, args.test_end)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"intervals": report["intervals"], "metrics": report["metrics"]}, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
