"""
statsbomb_data.py — Self-contained data bootstrap for the style-analytics scripts.

Downloads StatsBomb open data (FIFA World Cup 2022) on first run and writes:
  - sel_matches.json   : the selected match list
  - ev/<match_id>.json : per-match event data

Idempotent: anything already on disk is skipped, so the scripts run anywhere
(any machine, any folder) with no manual setup. StatsBomb open data is public
on GitHub, so no API key is needed.
"""
import os, json, urllib.request

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMP_ID, SEASON_ID = 43, 106  # FIFA World Cup 2022
TEAMS = {"Argentina", "France", "Brazil", "Morocco", "Spain", "Germany",
         "Portugal", "England", "Netherlands", "Croatia"}
MAX_MATCHES = 34
EV_DIR = "ev"
SEL_FILE = "sel_matches.json"


def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def ensure_data(verbose=True):
    """Make sure sel_matches.json and ev/*.json exist; download if missing."""
    os.makedirs(EV_DIR, exist_ok=True)

    # 1) selected match list
    if os.path.exists(SEL_FILE):
        sel = json.load(open(SEL_FILE))
    else:
        if verbose:
            print("Fetching WC 2022 match list from StatsBomb...")
        matches = _get_json(f"{BASE}/matches/{COMP_ID}/{SEASON_ID}.json")
        sel, seen = [], set()
        for m in matches:
            h = m["home_team"]["home_team_name"]
            a = m["away_team"]["away_team_name"]
            if (h in TEAMS or a in TEAMS) and m["match_id"] not in seen:
                seen.add(m["match_id"]); sel.append([m["match_id"], h, a])
        sel = sel[:MAX_MATCHES]
        json.dump(sel, open(SEL_FILE, "w"))

    # 2) per-match event files
    missing = [mid for mid, _, _ in sel if not os.path.exists(f"{EV_DIR}/{mid}.json")]
    if missing and verbose:
        print(f"Downloading event data for {len(missing)} matches "
              f"(~{len(missing)*3}MB, first run only)...")
    for mid in missing:
        urllib.request.urlretrieve(f"{BASE}/events/{mid}.json", f"{EV_DIR}/{mid}.json")

    if verbose:
        print(f"Data ready: {len(sel)} matches in ./{EV_DIR}/\n")
    return sel


if __name__ == "__main__":
    ensure_data()
