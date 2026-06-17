"""Style-clash (counter) analysis + visualization. Turn team style into
standardized vectors, build an interpretable "stylistic edge index" from
football logic, and draw a radar chart and a clash heatmap.

NOTE: this is an INTERPRETABLE HEURISTIC INDEX, not a win probability. Reliable
counter modeling needs a large sample of style-tagged matchups (see note at the
end); national teams play each other too rarely for that."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os, subprocess, sys
if not os.path.exists("team_style.csv"):
    print("team_style.csv not found - generating it first via analytics.py ...")
    subprocess.run([sys.executable, "analytics.py"], check=True)
style = pd.read_csv("team_style.csv", index_col="team", encoding="utf-8-sig")
teams = style.index.tolist()

# standardize (z-score)
metrics = ['possession', 'directness', 'long_ball', 'press_height', 'ppda',
           'field_tilt', 'xg_pg', 'setpiece_pct', 'counter_pct', 'crosses_pg']
Z = (style[metrics] - style[metrics].mean()) / style[metrics].std()

# --- derived style factors (interpretable) ---
# High-press intensity: high press location + low ppda (so negate ppda)
press = Z['press_height'] - Z['ppda']
# Build-up possession: high possession + short passing (negate directness) + high final-third touch
buildup = Z['possession'] - Z['directness'] + Z['field_tilt']
# Counter / transition threat: counter shot share + directness + long balls
counter = Z['counter_pct'] + Z['directness'] + Z['long_ball']
# Vulnerability in transition: high possession + high press line -> space left behind
trans_risk = Z['possession'] + Z['press_height']
# Set-piece threat
setp = Z['setpiece_pct'] + Z['crosses_pg']

factors = pd.DataFrame({'HighPress': press, 'BuildUp': buildup,
                        'CounterThreat': counter, 'TransitionRisk': trans_risk,
                        'SetPieces': setp})
factors.to_csv("team_factors.csv", encoding="utf-8-sig")

# --- edge index: A's stylistic advantage over B (higher = A exploits B's weaknesses) ---
def edge(A, B):
    # A's counter threat x B's transition risk
    e1 = counter[A] * trans_risk[B]
    # A's high press x B's reliance on building from the back (build-up high -> turnover-prone under press)
    e2 = press[A] * buildup[B]
    # A's set pieces x (B low possession -> usually more passive defensive setup; rough proxy)
    e3 = setp[A] * (-Z['possession'][B])
    return 0.5 * e1 + 0.35 * e2 + 0.15 * e3

M = pd.DataFrame([[edge(a, b) if a != b else np.nan for b in teams] for a in teams],
                 index=teams, columns=teams)
M.round(2).to_csv("matchup_matrix.csv", encoding="utf-8-sig")

# ================= visualization =================
fig = plt.figure(figsize=(15, 6.5), facecolor='white')

# (1) radar: style archetypes
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
radar_ax = ['possession', 'press_height', 'field_tilt', 'counter_pct', 'directness', 'xg_pg']
labels = ['Possession', 'Press\nHeight', 'Field\nTilt', 'Counter', 'Directness', 'xG/game']
R = (style[radar_ax]-style[radar_ax].min())/(style[radar_ax].max()-style[radar_ax].min())
ang = np.linspace(0, 2*np.pi, len(radar_ax), endpoint=False).tolist(); ang += ang[:1]
show = {'England': '#1f77b4', 'Argentina': '#2ca02c', 'Morocco': '#d62728', 'Japan': '#9467bd'}
for t, c in show.items():
    if t in R.index:
        v = R.loc[t].tolist(); v += v[:1]
        ax1.plot(ang, v, color=c, lw=2, label=t); ax1.fill(ang, v, color=c, alpha=0.08)
ax1.set_xticks(ang[:-1]); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_yticklabels([]); ax1.set_title("Team Style Fingerprints (WC 2022)", fontsize=12, pad=18)
ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9)

# (2) clash heatmap
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.imshow(M.values.astype(float), cmap='RdBu_r', vmin=-np.nanmax(abs(M.values)),
                vmax=np.nanmax(abs(M.values)))
ax2.set_xticks(range(len(teams))); ax2.set_yticks(range(len(teams)))
ax2.set_xticklabels(teams, rotation=45, ha='right', fontsize=8)
ax2.set_yticklabels(teams, fontsize=8)
ax2.set_xlabel("Opponent (B)", fontsize=10); ax2.set_ylabel("Team (A)", fontsize=10)
ax2.set_title("Stylistic Edge Index:  row A vs col B\n(red = A's style exploits B)", fontsize=11)
for i in range(len(teams)):
    for j in range(len(teams)):
        if not np.isnan(M.values[i, j]):
            ax2.text(j, i, f"{M.values[i, j]:.1f}", ha='center', va='center', fontsize=7,
                     color='white' if abs(M.values[i, j]) > np.nanmax(abs(M.values))*0.5 else 'black')
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("style_analysis.png", dpi=140, bbox_inches='tight')
print("saved style_analysis.png\n")

print("=== Style factors (z-score) ===")
print(factors.round(2).to_string())
print("\n=== Sample edge index (positive = row team's style counters column team) ===")
for a, b in [('Morocco', 'England'), ('Japan', 'Brazil'), ('England', 'Morocco'), ('Argentina', 'Japan')]:
    print(f"  {a:10s} vs {b:10s}: {M.loc[a, b]:+.2f}")
