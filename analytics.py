"""Quantify team playing style from StatsBomb event data (WC 2022 demo)."""
import os, json, math
import numpy as np, pandas as pd

GOAL = (120.0, 40.0)
def dist_goal(x, y): return math.hypot(GOAL[0]-x, GOAL[1]-y)

def load(mid): return json.load(open(f"ev/{mid}.json"))

sel = json.load(open("sel_matches.json"))
match_ids = [m[0] for m in sel]

# ---------- team style ----------
def team_match_stats(events):
    teams = list({e['team']['name'] for e in events if 'team' in e})
    out = {}
    for T in teams:
        te = [e for e in events if e.get('team', {}).get('name') == T]
        oe = [e for e in events if e.get('team', {}).get('name') != T and 'team' in e]
        passes = [e for e in te if e['type']['name'] == 'Pass']
        opp_passes = [e for e in oe if e['type']['name'] == 'Pass']
        comp = [p for p in passes if 'outcome' not in p.get('pass', {})]
        shots = [e for e in te if e['type']['name'] == 'Shot']
        pres = [e for e in te if e['type']['name'] == 'Pressure' and 'location' in e]
        touches = [e for e in te if 'location' in e and e['type']['name'] in
                   ('Pass', 'Ball Receipt*', 'Carry', 'Shot', 'Dribble', 'Duel', 'Clearance')]
        def_hi = [e for e in te if e['type']['name'] in
                  ('Pressure', 'Interception', 'Duel', 'Foul Committed') and
                  'location' in e and e['location'][0] >= 40]
        opp_build = [p for p in opp_passes if 'location' in p and p['location'][0] <= 80]
        npshots = [s for s in shots if s.get('shot', {}).get('type', {}).get('name') != 'Penalty']
        sp_shots = [s for s in shots if s.get('play_pattern', {}).get('name') in
                    ('From Corner', 'From Free Kick') or
                    s.get('shot', {}).get('type', {}).get('name') == 'Penalty']
        ctr_shots = [s for s in shots if s.get('play_pattern', {}).get('name') == 'From Counter']
        crosses = [p for p in passes if p.get('pass', {}).get('cross')]
        out[T] = dict(
            passes=len(passes), opp_passes=len(opp_passes), comp=len(comp),
            pass_len=np.mean([p['pass']['length'] for p in passes]) if passes else 0,
            long_pct=np.mean([p['pass']['length'] > 30 for p in passes]) if passes else 0,
            shots=len(shots), xg=sum(s.get('shot', {}).get('statsbomb_xg', 0) for s in shots),
            npxg=sum(s.get('shot', {}).get('statsbomb_xg', 0) for s in npshots),
            press_height=np.mean([p['location'][0] for p in pres]) if pres else 0,
            pressures=len(pres),
            ft_touch=np.mean([e['location'][0] > 80 for e in touches]) if touches else 0,
            sp_share=len(sp_shots)/len(shots) if shots else 0,
            ctr_share=len(ctr_shots)/len(shots) if shots else 0,
            crosses=len(crosses),
            ppda_num=len(opp_build), ppda_den=len(def_hi),
        )
    return out

agg = {}
for mid in match_ids:
    for T, s in team_match_stats(load(mid)).items():
        agg.setdefault(T, []).append(s)

rows = []
for T, lst in agg.items():
    if len(lst) < 3:  # too few matches to report
        continue
    d = pd.DataFrame(lst)
    rows.append(dict(
        team=T, games=len(lst),
        possession=100*d.passes.sum()/(d.passes.sum()+d.opp_passes.sum()),
        pass_cmp=100*d.comp.sum()/d.passes.sum(),
        directness=d.pass_len.mean(),
        long_ball=100*d.long_pct.mean(),
        press_height=d.press_height.mean(),
        ppda=d.ppda_num.sum()/max(d.ppda_den.sum(), 1),
        field_tilt=100*d.ft_touch.mean(),
        xg_pg=d.xg.mean(), shots_pg=d.shots.mean(),
        setpiece_pct=100*d.sp_share.mean(),
        counter_pct=100*d.ctr_share.mean(),
        crosses_pg=d.crosses.mean(),
    ))
style = pd.DataFrame(rows).set_index('team').round(1).sort_values('possession', ascending=False)
style.to_csv("team_style.csv")
pd.set_option('display.width', 200, 'display.max_columns', 30)
print("=== Team style fingerprints (WC 2022, per-match averages) ===")
print(style[['games', 'possession', 'directness', 'long_ball', 'press_height', 'ppda',
             'field_tilt', 'xg_pg', 'setpiece_pct', 'counter_pct']].to_string())
print("\nColumns: possession=possession %, directness=avg pass length (m), long_ball=long-pass %,")
print("press_height=avg location of pressures (higher = higher press), ppda=opp passes per defensive action (lower = more aggressive press),")
print("field_tilt=final-third touch %, xg_pg=xG per game, setpiece_pct=set-piece shot share, counter_pct=counter-attack shot share")
