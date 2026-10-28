"""Validated local result snapshots and a pinned, opt-in download."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import pandas as pd

SOURCE_COMMIT = "d5bd6de012c87e2f325de602ae77fe2f9ab61ef3"
SOURCE_URL = f"https://raw.githubusercontent.com/martj42/international_results/{SOURCE_COMMIT}/results.csv"
SOURCE_SHA256 = "5dcefd1aa0483979ea8378cef5be448d2005609d39945ea162378561a30a6767"
REQUIRED = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]


def load_results(path):
    frame = pd.read_csv(path)
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    if frame.date.isna().any():
        raise ValueError("Missing match date")
    for name in ["home_team", "away_team", "tournament"]:
        if frame[name].isna().any():
            raise ValueError(f"Missing {name}")
        frame[name] = frame[name].astype(str).str.strip()
        if frame[name].eq("").any():
            raise ValueError(f"Blank {name}")
    if frame.home_team.eq(frame.away_team).any():
        raise ValueError("A team cannot play itself")
    tokens = frame.neutral.astype(str).str.lower()
    if not tokens.isin(["true", "false"]).all():
        raise ValueError("neutral must contain only true or false")
    frame["neutral"] = tokens.eq("true")
    for name in ["home_score", "away_score"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        played = frame[name].dropna().to_numpy(dtype=float)
        if not np.isfinite(played).all() or (played < 0).any() or (played != np.floor(played)).any():
            raise ValueError("Scores must be finite nonnegative integers or both missing")
    if (frame.home_score.isna() != frame.away_score.isna()).any():
        raise ValueError("Both scores must be present or both missing")
    keys = ["date", "home_team", "away_team", "tournament"]
    # Exact duplicates add no evidence. Conflicting outcomes must be investigated.
    unique = frame.drop_duplicates(subset=REQUIRED).copy()
    unique.attrs["exact_duplicates_removed"] = len(frame) - len(unique)
    unique.attrs["conflicting_duplicate_rows"] = int(unique.duplicated(keys, keep=False).sum())
    return unique.sort_values(keys, kind="stable").reset_index(drop=True)


def assert_unique_matches(frame):
    if frame.duplicated(["date", "home_team", "away_team", "tournament"]).any():
        raise ValueError("Conflicting duplicate match identity in selected interval")


def ensure_data(path="results.csv", refresh=False):
    path = Path(path)
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".csv", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
        if sha256(temporary) != SOURCE_SHA256:
            raise ValueError("Pinned source checksum mismatch; review the download before using it")
        load_results(temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
