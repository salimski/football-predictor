"""
Team name normalization.

Two sources use different names for the same clubs:
  - api-sports.io  : some seasons use German names (sometimes with corrupted
                     UTF-8, U+FFFD replacement char), others use English names,
                     and some clubs have prefix/suffix variations.
  - Understat      : uses short English names with no articles.

This module defines a single canonical name per club (correct German spelling)
and provides a normalize() helper used by all downstream code.

Running this script directly patches the database:
  1. Adds home_team / away_team columns to the matches table and populates
     them with canonical names extracted from raw_json.
  2. Updates xg.home_team / xg.away_team to canonical names.
  3. Re-runs the join test and prints before/after counts.
"""

import sqlite3
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH

# ---------------------------------------------------------------------------
# Canonical mapping
# Keys  : every known variant from api-sports.io AND Understat
# Values: single canonical name (correct German spelling)
# ---------------------------------------------------------------------------

TEAM_NAME_MAP: dict[str, str] = {
    # --- Bayern Munich ---
    "Bayern Munich":                "Bayern Munich",       # api-sports.io (English)
    "Bayern München":              "Bayern Munich",       # api-sports.io (German, ü=U+00FC)
    "Bayern M\uFFFDnchen":         "Bayern Munich",       # api-sports.io (corrupted, just in case)

    # --- Borussia Mönchengladbach ---
    "Borussia Mönchengladbach":    "Borussia Mönchengladbach",    # ö=U+00F6
    "Borussia M\uFFFDnchengladbach": "Borussia Mönchengladbach",  # corrupted, just in case
    "Borussia Monchengladbach":    "Borussia Mönchengladbach",    # api-sports.io (no umlaut)
    "Borussia M.Gladbach":         "Borussia Mönchengladbach",    # Understat

    # --- 1. FC Köln ---
    "1. FC Köln":                  "1. FC Köln",                  # ö=U+00F6
    "1. FC K\uFFFDln":             "1. FC Köln",                  # corrupted, just in case
    "FC Cologne":                  "1. FC Köln",                  # Understat

    # --- VfL Bochum ---
    "VfL Bochum":                  "VfL Bochum",
    "Vfl Bochum":                  "VfL Bochum",     # api-sports.io (wrong capitalisation)
    "Bochum":                      "VfL Bochum",     # Understat

    # --- 1. FC Heidenheim ---
    "1. FC Heidenheim":            "1. FC Heidenheim",
    "FC Heidenheim":               "1. FC Heidenheim",    # Understat

    # --- SpVgg Greuther Fürth ---
    "SpVgg Greuther Furth":        "SpVgg Greuther Fürth",   # api-sports.io (no umlaut)
    "SpVgg Greuther Fürth":        "SpVgg Greuther Fürth",   # ü=U+00FC
    "Greuther Fuerth":             "SpVgg Greuther Fürth",   # Understat

    # --- RB Leipzig ---
    "RB Leipzig":                  "RB Leipzig",
    "RasenBallsport Leipzig":      "RB Leipzig",    # Understat

    # --- 1899 Hoffenheim ---
    "1899 Hoffenheim":             "1899 Hoffenheim",
    "Hoffenheim":                  "1899 Hoffenheim",    # Understat

    # --- FSV Mainz 05 ---
    "FSV Mainz 05":                "FSV Mainz 05",
    "Mainz 05":                    "FSV Mainz 05",    # Understat

    # --- FC Augsburg ---
    "FC Augsburg":                 "FC Augsburg",
    "Augsburg":                    "FC Augsburg",    # Understat

    # --- SC Freiburg ---
    "SC Freiburg":                 "SC Freiburg",
    "Freiburg":                    "SC Freiburg",    # Understat

    # --- SV Darmstadt 98 ---
    "SV Darmstadt 98":             "SV Darmstadt 98",
    "Darmstadt":                   "SV Darmstadt 98",    # Understat

    # --- VfL Wolfsburg ---
    "VfL Wolfsburg":               "VfL Wolfsburg",
    "Wolfsburg":                   "VfL Wolfsburg",    # Understat

    # --- FC Schalke 04 ---
    "FC Schalke 04":               "FC Schalke 04",
    "Schalke 04":                  "FC Schalke 04",    # Understat

    # --- FC St. Pauli ---
    "FC St. Pauli":                "FC St. Pauli",
    "St. Pauli":                   "FC St. Pauli",    # Understat

    # --- Relegation playoff opponents (not regular Bundesliga members) ---
    "Fortuna Dusseldorf":          "Fortuna Düsseldorf",
    "Fortuna Düsseldorf":          "Fortuna Düsseldorf",

    # --- Identical in both sources ---
    "Arminia Bielefeld":           "Arminia Bielefeld",
    "Bayer Leverkusen":            "Bayer Leverkusen",
    "Borussia Dortmund":           "Borussia Dortmund",
    "Eintracht Frankfurt":         "Eintracht Frankfurt",
    "Hamburger SV":                "Hamburger SV",
    "Hertha Berlin":               "Hertha Berlin",
    "Holstein Kiel":               "Holstein Kiel",
    "SV Elversberg":               "SV Elversberg",
    "Union Berlin":                "Union Berlin",
    "VfB Stuttgart":               "VfB Stuttgart",
    "Werder Bremen":               "Werder Bremen",

    # --- football-data.co.uk short names ---
    "Bielefeld":                   "Arminia Bielefeld",
    "Dortmund":                    "Borussia Dortmund",
    "Ein Frankfurt":               "Eintracht Frankfurt",
    "FC Koln":                     "1. FC Köln",
    "Greuther Furth":              "SpVgg Greuther Fürth",
    "Hamburg":                     "Hamburger SV",
    "Heidenheim":                  "1. FC Heidenheim",
    "Hertha":                      "Hertha Berlin",
    "Hoffenheim":                  "1899 Hoffenheim",          # also Understat
    "Leverkusen":                  "Bayer Leverkusen",
    "M'gladbach":                  "Borussia Mönchengladbach",
    "Mainz":                       "FSV Mainz 05",
    "St Pauli":                    "FC St. Pauli",
    "Stuttgart":                   "VfB Stuttgart",
}


def normalize(name: str) -> str:
    """Return the canonical team name for any known variant. Falls back to the
    original name so unknown entries are visible rather than silently dropped."""
    return TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Database patching
# ---------------------------------------------------------------------------

def _join_count(conn) -> int:
    """Count how many matches rows successfully join to xg on date + team names."""
    return conn.execute("""
        SELECT COUNT(*) FROM matches m
        JOIN xg x
          ON substr(json_extract(m.raw_json,'$.fixture.date'),1,10) = x.match_date
         AND m.home_team = x.home_team
         AND m.away_team = x.away_team
    """).fetchone()[0]


def patch_matches(conn):
    """Add home_team / away_team columns to matches and fill with canonical names."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
    if "home_team" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN home_team TEXT")
    if "away_team" not in cols:
        conn.execute("ALTER TABLE matches ADD COLUMN away_team TEXT")

    rows = conn.execute("SELECT match_id, raw_json FROM matches").fetchall()
    updated = 0
    unknown = set()

    for match_id, raw in rows:
        fx = json.loads(raw)
        raw_home = fx.get("teams", {}).get("home", {}).get("name", "")
        raw_away = fx.get("teams", {}).get("away", {}).get("name", "")
        norm_home = normalize(raw_home)
        norm_away = normalize(raw_away)
        if norm_home == raw_home and raw_home not in TEAM_NAME_MAP:
            unknown.add(raw_home)
        if norm_away == raw_away and raw_away not in TEAM_NAME_MAP:
            unknown.add(raw_away)
        conn.execute(
            "UPDATE matches SET home_team = ?, away_team = ? WHERE match_id = ?",
            (norm_home, norm_away, match_id),
        )
        updated += 1

    conn.commit()
    print(f"  matches: set home_team/away_team for {updated} rows")
    if unknown:
        print(f"  WARNING — unmapped names (kept as-is): {sorted(unknown)}")


def patch_xg(conn):
    """Normalise home_team / away_team in the xg table in place."""
    rows = conn.execute("SELECT id, home_team, away_team FROM xg").fetchall()
    updated = 0
    unknown = set()

    for row_id, raw_home, raw_away in rows:
        norm_home = normalize(raw_home)
        norm_away = normalize(raw_away)
        if norm_home == raw_home and raw_home not in TEAM_NAME_MAP:
            unknown.add(raw_home)
        if norm_away == raw_away and raw_away not in TEAM_NAME_MAP:
            unknown.add(raw_away)
        conn.execute(
            "UPDATE xg SET home_team = ?, away_team = ? WHERE id = ?",
            (norm_home, norm_away, row_id),
        )
        updated += 1

    conn.commit()
    print(f"  xg: normalised {updated} rows")
    if unknown:
        print(f"  WARNING — unmapped names (kept as-is): {sorted(unknown)}")


def run():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # Baseline (uses raw json_extract — before columns exist)
    before = conn.execute("""
        SELECT COUNT(*) FROM matches m
        JOIN xg x
          ON substr(json_extract(m.raw_json,'$.fixture.date'),1,10) = x.match_date
         AND json_extract(m.raw_json,'$.teams.home.name') = x.home_team
         AND json_extract(m.raw_json,'$.teams.away.name') = x.away_team
    """).fetchone()[0]
    print(f"Join count BEFORE: {before} / 1456  ({before/1456*100:.1f}%)")

    print("\nPatching matches table...")
    patch_matches(conn)

    print("Patching xg table...")
    patch_xg(conn)

    after = _join_count(conn)
    print(f"\nJoin count AFTER:  {after} / 1456  ({after/1456*100:.1f}%)")

    # Distinct canonical team names in matches
    n_teams = conn.execute(
        "SELECT COUNT(DISTINCT home_team) FROM matches"
    ).fetchone()[0]
    print(f"Distinct team names in matches: {n_teams}  (expected 18)")

    conn.close()


if __name__ == "__main__":
    run()
