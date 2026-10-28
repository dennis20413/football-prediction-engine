import io
import itertools
import json
import runpy
import urllib.error

import numpy as np
import pandas as pd
import pytest

import data_contract
from backtest import evaluate, main, metrics, reliability, split_matches
from data_contract import load_results
from predictor import MatchPredictor, effective_rho
from strength import StrengthModel
from tournament import Tournament, build_groups


def matches():
    rows = []
    for i in range(20):
        rows.append(dict(date=f"2022-01-{i+1:02}", home_team="A", away_team="B",
                         home_score=i % 4, away_score=(i + 1) % 3,
                         tournament="Friendly", neutral=i % 2 == 0))
    rows.extend([dict(rows[0], date="2024-01-01", home_score=1, away_score=0),
                 dict(rows[1], date="2024-01-02", home_score=0, away_score=1)])
    return pd.DataFrame(rows)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "results.csv"
    matches().to_csv(path, index=False)
    return path


def test_baseline_accuracy_uses_test_outcomes():
    prior = np.tile([.7, .2, .1], (3, 1))
    result = metrics(prior, np.array([0, 1, 2]))
    assert result["accuracy"] == pytest.approx(1/3)
    assert result["accuracy"] != .7
    assert result["brier_multiclass_sum"] == pytest.approx(np.square(prior-np.eye(3)).sum(axis=1).mean())


@pytest.mark.parametrize("p", [[], [[.2, .2, .2]], [[-.1, .1, 1]], [[float('nan'), 0, 1]]])
def test_invalid_probabilities_fail(p):
    with pytest.raises(ValueError):
        metrics(p, [0] * len(p))


def test_reliability_includes_shared_edges_and_one():
    p = np.array([0, .2, .4, .6, .8, 1])
    result = reliability(p, np.zeros(6))
    assert sum(row["matches"] for row in result) == 6
    assert result[-1]["matches"] == 2


def test_interval_boundaries_are_disjoint(csv_path):
    train, test = split_matches(load_results(csv_path), "2024-01-01", "2024-01-02")
    assert train.date.max() < pd.Timestamp("2024-01-01")
    assert test.date.tolist() == [pd.Timestamp("2024-01-01")]


def test_test_scores_cannot_change_fitted_model(csv_path):
    original = MatchPredictor(cutoff="2024-01-01").fit(csv_path)
    changed = matches()
    changed.loc[changed.date >= "2024-01-01", ["home_score", "away_score"]] = [20, 30]
    changed.to_csv(csv_path, index=False)
    altered = MatchPredictor(cutoff="2024-01-01").fit(csv_path)
    np.testing.assert_allclose(original.model.model.coef_, altered.model.model.coef_, atol=0, rtol=0)
    assert original.model.model.intercept_ == altered.model.model.intercept_
    assert original.rho == altered.rho


def test_report_has_measured_baseline_coverage_and_hash(csv_path):
    report = evaluate(csv_path, test_end="2024-02-01")
    assert report["intervals"]["evaluated_test_matches"] == 2
    assert report["metrics"]["same_window_prior"]["accuracy"] == .5
    assert report["source"]["sha256"] == data_contract.sha256(csv_path)
    assert sum(row["matches"] for row in report["home_win_reliability"]) == 2


def test_all_unseen_teams_fail(csv_path):
    frame = matches()
    frame.loc[frame.date >= "2024-01-01", "home_team"] = "Unseen"
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="represented"):
        evaluate(csv_path, test_end="2024-02-01")


def test_import_never_downloads(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Import must be offline")
    monkeypatch.setattr(data_contract.urllib.request, "urlopen", fail)
    runpy.run_module("backtest", run_name="import_check")


@pytest.mark.parametrize("field,value", [("neutral", "maybe"), ("home_score", -1),
                                          ("home_score", 1.5), ("home_score", float('inf')),
                                          ("date", "bad"), ("home_team", "")])
def test_invalid_inputs(csv_path, field, value):
    frame = matches().astype({field: object})
    frame.loc[0, field] = value
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_results(csv_path)


def test_missing_scores_must_be_paired(csv_path):
    frame = matches()
    frame.loc[0, "home_score"] = np.nan
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Both scores"):
        load_results(csv_path)
    frame.loc[0, "away_score"] = np.nan
    frame.to_csv(csv_path, index=False)
    assert load_results(csv_path).home_score.isna().sum() == 1


def test_conflicting_duplicates_rejected_only_in_selected_interval(csv_path):
    frame = matches()
    duplicate = dict(frame.iloc[0], home_score=40)
    pd.concat([frame, pd.DataFrame([duplicate])]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        split_matches(load_results(csv_path), "2024-01-01", "2024-02-01")
    frame.loc[0, "date"] = "1900-01-01"
    duplicate["date"] = "1900-01-01"
    pd.concat([frame, pd.DataFrame([duplicate])]).to_csv(csv_path, index=False)
    train, _ = split_matches(load_results(csv_path), "2024-01-01", "2024-02-01")
    assert len(train) == 19


def test_output_cannot_replace_input(csv_path):
    before = csv_path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--data", str(csv_path), "--output", str(csv_path)])
    assert error.value.code == 2 and csv_path.read_bytes() == before


def test_failed_refresh_preserves_prior_file(csv_path, monkeypatch):
    before = csv_path.read_bytes()
    def fail(*args, **kwargs):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(data_contract.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        data_contract.ensure_data(csv_path, refresh=True)
    assert csv_path.read_bytes() == before


@pytest.mark.parametrize("rates,rho", [((12, 12), -.2), ((.1, 8), .2), ((1.2, 1.5), -.05)])
def test_score_grid_is_valid_and_covers_tail(rates, rho):
    class Model:
        def expected_goals(self, *args, **kwargs):
            return rates
    pred = MatchPredictor()
    pred.model, pred.rho = Model(), rho
    matrix, _, _ = pred.score_matrix("A", "B")
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1)
    report = pred.predict("A", "B")
    assert sum(report[n] for n in ["p_home", "p_draw", "p_away"]) == pytest.approx(1)
    if max(rates) == 12:
        assert len(matrix) > 12


@pytest.mark.parametrize("rate", [0, -1, np.inf, 1e-200, 101])
def test_unsupported_rates_fail(rate):
    with pytest.raises(ValueError):
        effective_rho(rate, 1, -.1)


def group_fixtures():
    rows = []
    for group in range(12):
        for a, b in itertools.combinations(range(4), 2):
            rows.append(dict(home_team=f"G{group}T{a}", away_team=f"G{group}T{b}",
                             home_score=np.nan, away_score=np.nan))
    return pd.DataFrame(rows)


def test_tournament_rejects_knockout_connected_input():
    frame = group_fixtures()
    assert len(build_groups(frame)) == 12
    knockout = dict(home_team="G0T0", away_team="G1T0", home_score=1, away_score=0)
    with pytest.raises(ValueError, match="group-stage"):
        build_groups(pd.concat([frame, pd.DataFrame([knockout])]))


def test_simplified_tournament_seed_is_reproducible():
    class Model:
        def expected_goals(self, *args, **kwargs):
            return 1.3, 1.3
    tour = Tournament(Model(), group_fixtures())
    pd.testing.assert_frame_equal(tour.monte_carlo(20, 42), tour.monte_carlo(20, 42))
    with pytest.raises(ValueError):
        tour.monte_carlo(0)


def test_strength_rejects_unfitted_unknown_and_self(csv_path):
    model = StrengthModel()
    with pytest.raises(RuntimeError):
        model.expected_goals("A", "B")
    model.fit(csv_path, "2024-01-01")
    with pytest.raises(ValueError):
        model.expected_goals("A", "Unknown")
    with pytest.raises(ValueError):
        model.expected_goals("A", "A")
