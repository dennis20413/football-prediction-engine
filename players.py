"""Per-90 player profiles: attacking output + progression + defensive work
(WC 2022, from StatsBomb event data)."""
import os, json, math
import numpy as np, pandas as pd

GOAL = (120.0, 40.0)
def d2g(loc): return math.hypot(GOAL[0]-loc[0], GOAL[1]-loc[1])
sel = json.load(open("sel_matches.json")); mids = [m[0] for m in sel]

def minutes(events):
    """Return {player: minutes played}."""
    end = {}
    for e in events:  # max minute reached in each period
        end[e['period']] = max(end.get(e['period'], 0), e.get('minute', 0))
    match_end = sum(end.get(p, 0) for p in end)  # approximate total duration
    on = {}; off = {}
    for e in events:
        if e['type']['name'] == 'Starting XI':
            for p in e['tactics']['lineup']:
                on[p['player']['name']] = 0
        elif e['type']['name'] == 'Substitution':
            off[e['player']['name']] = e['minute']
            on[e['substitution']['replacement']['name']] = e['minute']
    mins = {}
    for p in on:
        mins[p] = max(off.get(p, match_end) - on[p], 0)
    return mins

stat = {}; mins_tot = {}
for mid in mids:
    ev = json.load(open(f"ev/{mid}.json"))
    by_id = {e['id']: e for e in ev}
    for p, m in minutes(ev).items():
        mins_tot[p] = mins_tot.get(p, 0) + m
    for e in ev:
        p = e.get('player', {}).get('name')
        if not p: continue
        s = stat.setdefault(p, dict(npxg=0, xa=0, shots=0, kp=0, prog_pass=0,
                                    carries_prog=0, drib=0, defact=0))
        t = e['type']['name']
        if t == 'Shot':
            sh = e['shot']
            if sh.get('type', {}).get('name') != 'Penalty':
                s['npxg'] += sh.get('statsbomb_xg', 0)
            s['shots'] += 1
        if t == 'Pass':
            pa = e['pass']
            if pa.get('shot_assist') or pa.get('goal_assist'): s['kp'] += 1
            if 'outcome' not in pa and 'location' in e and 'end_location' in pa:
                if d2g(e['location']) - d2g(pa['end_location']) >= 10: s['prog_pass'] += 1
        if t == 'Carry' and 'location' in e:
            el = e.get('carry', {}).get('end_location')
            if el and d2g(e['location']) - d2g(el) >= 5: s['carries_prog'] += 1
        if t == 'Dribble' and e.get('dribble', {}).get('outcome', {}).get('name') == 'Complete':
            s['drib'] += 1
        if t in ('Pressure', 'Ball Recovery', 'Interception', 'Block', 'Clearance'):
            s['defact'] += 1
        if t == 'Duel' and e.get('duel', {}).get('outcome', {}).get('name') in ('Won', 'Success In Play'):
            s['defact'] += 1
    # xA: the key-pass player for a shot gets that shot's xG
    for e in ev:
        if e['type']['name'] == 'Shot' and 'key_pass_id' in e['shot']:
            kp = by_id.get(e['shot']['key_pass_id'])
            if kp and kp.get('player'):
                stat[kp['player']['name']]['xa'] += e['shot'].get('statsbomb_xg', 0)

def per90(name):
    m = mins_tot.get(name, 0)
    if m < 200: return None
    s = stat[name]; f = 90 / m
    return dict(player=name, mins=int(m),
                npxG=round(s['npxg']*f, 2), xA=round(s['xa']*f, 2),
                shots=round(s['shots']*f, 1), key_passes=round(s['kp']*f, 1),
                prog_passes=round(s['prog_pass']*f, 1), prog_carries=round(s['carries_prog']*f, 1),
                dribbles=round(s['drib']*f, 1), def_actions=round(s['defact']*f, 1))

players = ["Lionel Andrés Messi Cuccittini", "Kylian Mbappé Lottin",
           "Antoine Griezmann", "Julián Álvarez", "Sofyan Amrabat",
           "Jude Bellingham", "Bukayo Saka", "Cody Gakpo", "Luka Modrić"]
print("=== Player per-90 profiles (WC 2022) ===")
rows = [per90(p) for p in players]; rows = [r for r in rows if r]
df = pd.DataFrame(rows)
df['player'] = df['player'].str.split().str[-1]  # short surname for display
print(df.to_string(index=False))
print("\nnpxG=non-penalty xG, xA=expected assists, prog=progressive,")
print("def_actions=pressures+recoveries+interceptions+blocks+clearances+duels won (per 90)")
df.to_csv("player_profiles.csv", index=False)
