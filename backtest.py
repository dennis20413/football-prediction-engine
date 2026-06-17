"""
backtest.py — Honestly validate model accuracy (out-of-time sample).
Train on data before TRAIN_CUT, predict every international match in
[TRAIN_CUT, TEST_END], and report log-loss / Brier / accuracy / goal error,
compared against a base-rate baseline.
"""
import numpy as np, pandas as pd
from predictor import MatchPredictor, ensure_data, DATA

TRAIN_CUT = "2024-01-01"
TEST_END = "2026-06-11"

ensure_data(refresh=False)
print(f"Train: <{TRAIN_CUT}   Test: {TRAIN_CUT} ~ {TEST_END}")
pred = MatchPredictor(cutoff=TRAIN_CUT).fit()
print(f"Dixon-Coles rho = {pred.rho:.3f}\n")

df = pd.read_csv(DATA, parse_dates=["date"])
test = df[(df.date >= TRAIN_CUT) & (df.date < TEST_END)].dropna(subset=["home_score", "away_score"])
ok = test.home_team.isin(pred.model.teams) & test.away_team.isin(pred.model.teams)
test = test[ok].reset_index(drop=True)

# Base rates from the training period (constant-prediction baseline)
tr = df[df.date < TRAIN_CUT].dropna(subset=["home_score", "away_score"])
def outcome(hs, as_): return 0 if hs > as_ else (1 if hs == as_ else 2)  # H / D / A
base = np.bincount([outcome(h, a) for h, a in zip(tr.home_score, tr.away_score)], minlength=3)
base = base / base.sum()

rows = []
for _, m in test.iterrows():
    r = pred.predict(m.home_team, m.away_team, neutral=bool(m.neutral))
    p = np.array([r["p_home"], r["p_draw"], r["p_away"]])
    y = outcome(m.home_score, m.away_score)
    rows.append((p, y, r["xg_home"] + r["xg_away"], m.home_score + m.away_score))

P = np.array([r[0] for r in rows]); Y = np.array([r[1] for r in rows])
pred_tot = np.array([r[2] for r in rows]); act_tot = np.array([r[3] for r in rows])
onehot = np.eye(3)[Y]

def logloss(P): return -np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)))
def brier(P): return np.mean(np.sum((P - onehot) ** 2, axis=1))
acc = np.mean(P.argmax(1) == Y)
base_P = np.tile(base, (len(Y), 1))

print(f"Test matches: {len(Y)}")
print(f"Actual outcomes  Home {np.mean(Y==0)*100:.0f}% / Draw {np.mean(Y==1)*100:.0f}% / Away {np.mean(Y==2)*100:.0f}%\n")
print(f"{'Metric':<14}{'Model':>10}{'Baseline':>12}   (lower is better; accuracy higher is better)")
print(f"{'Log-loss':<14}{logloss(P):>10.4f}{logloss(base_P):>12.4f}")
print(f"{'Brier':<14}{brier(P):>10.4f}{brier(base_P):>12.4f}")
print(f"{'Accuracy':<14}{acc*100:>9.1f}%{max(base)*100:>11.1f}%")
print(f"{'Goals MAE':<14}{np.mean(np.abs(pred_tot-act_tot)):>10.2f}{'-':>12}")

# Calibration: bin predicted home-win prob, compare to observed home-win rate
print("\nCalibration check (predicted home-win prob vs actual home-win rate):")
ph = P[:, 0]; home_win = (Y == 0).astype(int)
for lo in [0, .2, .4, .6, .8]:
    mask = (ph >= lo) & (ph < lo + .2)
    if mask.sum() > 5:
        print(f"  predicted {lo:.0%}-{lo+.2:.0%}: actual home win {home_win[mask].mean()*100:4.0f}%  (n={mask.sum()})")
