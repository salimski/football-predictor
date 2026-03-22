"""
Incremental data update — fetches only NEW matches since last collection.

Updates all tables:
  1. matches, statistics, lineups  (api-sports.io — only new fixtures)
  2. standings                     (api-sports.io — current season only)
  3. xg                            (Understat — current season, INSERT OR REPLACE)
  4. understat_player_stats        (Understat — current season)
  5. understat_formation_stats     (Understat — current season)
  6. odds                          (football-data.co.uk CSV — re-import current season)
  7. features                      (rebuild from raw data)

Usage:
    python update_data.py
"""

import json
import sqlite3
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    API_KEY, API_HOST, BASE_URL, LEAGUE_ID,
    DB_PATH, REQUEST_DELAY, SEASONS, UNDERSTAT_SEASONS,
)
from collector.apifootball import api_get, get_conn, init_db
from collector.understat import fetch_season_xg, store_xg, init_xg_table
from collector.understat_advanced import (
    collect_player_stats as collect_understat_players,
    collect_formation_stats as collect_understat_formations,
    init_player_stats_table, init_formation_stats_table,
    LABEL_MAP,
)
from collector.normalize import normalize


CURRENT_SEASON = SEASONS[-1]  # 2025/26
CURRENT_YEAR = CURRENT_SEASON["from"][:4]  # "2025"


def get_last_match_date(conn):
    """Get the date of the most recent match in the DB."""
    row = conn.execute(
        "SELECT MAX(json_extract(raw_json, '$.fixture.date')) FROM matches"
    ).fetchone()
    if row and row[0]:
        return row[0][:10]  # YYYY-MM-DD
    return None


def get_existing_match_ids(conn):
    """Get all match IDs already in the DB."""
    rows = conn.execute("SELECT match_id FROM matches").fetchall()
    return {r[0] for r in rows}


def get_match_ids_with_stats(conn):
    """Get match IDs that already have statistics."""
    rows = conn.execute("SELECT match_id FROM statistics").fetchall()
    return {r[0] for r in rows}


def get_match_ids_with_lineups(conn):
    """Get match IDs that already have lineups."""
    rows = conn.execute("SELECT match_id FROM lineups").fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Step 1: Fetch new fixtures from api-sports.io
# ---------------------------------------------------------------------------

def update_fixtures(conn):
    """Fetch only finished matches not yet in the DB."""
    last_date = get_last_match_date(conn)
    existing_ids = get_existing_match_ids(conn)

    print(f"\n{'='*50}")
    print(f"Step 1: Fetch new fixtures (after {last_date})")
    print(f"{'='*50}")
    print(f"  Existing matches in DB: {len(existing_ids)}")

    # Fetch all finished matches for current season
    params = {"league": LEAGUE_ID, "season": CURRENT_YEAR, "status": "FT"}
    data = api_get("/fixtures", params)
    time.sleep(REQUEST_DELAY)

    if not data:
        print("  No data returned from API")
        return []

    new_ids = []
    for fixture in data:
        match_id = str(fixture.get("fixture", {}).get("id", ""))
        if not match_id or match_id in existing_ids:
            continue

        match_date = fixture.get("fixture", {}).get("date", "")[:10]
        home = fixture.get("teams", {}).get("home", {}).get("name", "?")
        away = fixture.get("teams", {}).get("away", {}).get("name", "?")

        try:
            conn.execute(
                "INSERT OR REPLACE INTO matches (match_id, season, raw_json) VALUES (?, ?, ?)",
                (match_id, CURRENT_SEASON["label"], json.dumps(fixture)),
            )
            new_ids.append(match_id)
            print(f"  NEW: {match_date} {home} vs {away} (ID: {match_id})")
        except Exception as exc:
            print(f"  [db error] {match_id}: {exc}")

    conn.commit()
    print(f"\n  {len(new_ids)} new matches added")
    return new_ids


# ---------------------------------------------------------------------------
# Step 2: Fetch statistics & lineups for new matches
# ---------------------------------------------------------------------------

def update_statistics(conn, new_match_ids):
    """Fetch statistics for matches that don't have them yet."""
    have_stats = get_match_ids_with_stats(conn)
    need_stats = [mid for mid in new_match_ids if mid not in have_stats]

    # Also check for any older matches missing stats
    all_ids = get_existing_match_ids(conn)
    missing_old = [mid for mid in all_ids if mid not in have_stats and mid not in need_stats]
    if missing_old:
        print(f"  Also found {len(missing_old)} older matches missing statistics")
        need_stats.extend(missing_old[:20])  # cap at 20 to save API calls

    print(f"\n{'='*50}")
    print(f"Step 2a: Fetch statistics ({len(need_stats)} matches)")
    print(f"{'='*50}")

    inserted = 0
    for i, match_id in enumerate(need_stats, 1):
        print(f"  [{i}/{len(need_stats)}] fixture {match_id}", end="")
        data = api_get("/fixtures/statistics", {"fixture": match_id})
        time.sleep(REQUEST_DELAY)

        if not data:
            print(" — no data")
            continue

        try:
            conn.execute(
                "INSERT OR REPLACE INTO statistics (match_id, season, raw_json) VALUES (?, ?, ?)",
                (match_id, CURRENT_SEASON["label"], json.dumps(data)),
            )
            inserted += 1
            print(" — OK")
        except Exception as exc:
            print(f" — error: {exc}")

    conn.commit()
    print(f"  {inserted} statistics rows added")
    return inserted


def update_lineups(conn, new_match_ids):
    """Fetch lineups for matches that don't have them yet."""
    have_lineups = get_match_ids_with_lineups(conn)
    need_lineups = [mid for mid in new_match_ids if mid not in have_lineups]

    # Also check for older matches missing lineups
    all_ids = get_existing_match_ids(conn)
    missing_old = [mid for mid in all_ids if mid not in have_lineups and mid not in need_lineups]
    if missing_old:
        print(f"  Also found {len(missing_old)} older matches missing lineups")
        need_lineups.extend(missing_old[:20])

    print(f"\n{'='*50}")
    print(f"Step 2b: Fetch lineups ({len(need_lineups)} matches)")
    print(f"{'='*50}")

    inserted = 0
    for i, match_id in enumerate(need_lineups, 1):
        print(f"  [{i}/{len(need_lineups)}] fixture {match_id}", end="")
        data = api_get("/fixtures/lineups", {"fixture": match_id})
        time.sleep(REQUEST_DELAY)

        if not data:
            print(" — no data")
            continue

        try:
            conn.execute(
                "INSERT OR REPLACE INTO lineups (match_id, season, raw_json) VALUES (?, ?, ?)",
                (match_id, CURRENT_SEASON["label"], json.dumps(data)),
            )
            inserted += 1
            print(" — OK")
        except Exception as exc:
            print(f" — error: {exc}")

    conn.commit()
    print(f"  {inserted} lineups rows added")
    return inserted


# ---------------------------------------------------------------------------
# Step 3: Update standings (current season only)
# ---------------------------------------------------------------------------

def update_standings(conn):
    print(f"\n{'='*50}")
    print(f"Step 3: Update standings ({CURRENT_SEASON['label']})")
    print(f"{'='*50}")

    data = api_get("/standings", {"league": LEAGUE_ID, "season": CURRENT_YEAR})
    time.sleep(REQUEST_DELAY)

    if not data:
        print("  No standings data")
        return 0

    conn.execute(
        "INSERT OR REPLACE INTO standings (season, raw_json) VALUES (?, ?)",
        (CURRENT_SEASON["label"], json.dumps(data)),
    )
    conn.commit()
    print("  Standings updated")
    return 1


# ---------------------------------------------------------------------------
# Step 4: Update Understat xG (current season — full re-scrape)
# ---------------------------------------------------------------------------

def update_xg(conn):
    print(f"\n{'='*50}")
    print(f"Step 4: Update Understat xG ({CURRENT_SEASON['label']})")
    print(f"{'='*50}")

    init_xg_table(conn)

    try:
        matches = fetch_season_xg(CURRENT_YEAR)
        count = store_xg(conn, CURRENT_SEASON["label"], matches)
        print(f"  {count} xG rows updated")
        return count
    except Exception as exc:
        print(f"  [error] {exc}")
        return 0


# ---------------------------------------------------------------------------
# Step 5: Update Understat player & formation stats (current season)
# ---------------------------------------------------------------------------

def update_understat_advanced(conn):
    print(f"\n{'='*50}")
    print(f"Step 5: Update Understat player & formation stats")
    print(f"{'='*50}")

    # Player stats — only current season
    import collector.understat_advanced as ua
    original_seasons = ua.UNDERSTAT_SEASONS

    # Temporarily override to only fetch current season
    ua.UNDERSTAT_SEASONS = [CURRENT_YEAR]
    try:
        init_player_stats_table(conn)
        player_count = collect_understat_players(conn)
        print(f"  Player stats: {player_count} rows")

        init_formation_stats_table(conn)
        formation_count = collect_understat_formations(conn)
        print(f"  Formation stats: {formation_count} rows")
    finally:
        ua.UNDERSTAT_SEASONS = original_seasons

    return player_count, formation_count


# ---------------------------------------------------------------------------
# Step 6: Update odds (re-import current season CSV)
# ---------------------------------------------------------------------------

def update_odds(conn):
    print(f"\n{'='*50}")
    print(f"Step 6: Update odds from football-data.co.uk CSVs")
    print(f"{'='*50}")

    from collector.odds_collector import load_csv

    csv_dir = os.path.join("data", "external", "football_data_co_uk")
    filename = "25_26(1).csv"
    filepath = os.path.join(csv_dir, filename)

    if not os.path.exists(filepath):
        print(f"  CSV not found: {filepath}")
        print("  Download latest from football-data.co.uk if needed")
        return 0

    # Delete current season odds and re-import
    conn.execute("DELETE FROM odds WHERE season = ?", (CURRENT_SEASON["label"],))
    conn.commit()

    df = load_csv(filepath, CURRENT_SEASON["label"])
    if len(df) == 0:
        print("  No odds rows found in CSV")
        return 0

    df.to_sql("odds", conn, if_exists="append", index=False)
    conn.commit()
    print(f"  {len(df)} odds rows imported for {CURRENT_SEASON['label']}")
    return len(df)


# ---------------------------------------------------------------------------
# Step 7: Rebuild features
# ---------------------------------------------------------------------------

def rebuild_features():
    print(f"\n{'='*50}")
    print(f"Step 7: Rebuild features table")
    print(f"{'='*50}")

    import subprocess
    result = subprocess.run(
        [sys.executable, "build_features.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  [error] {result.stderr}")
    else:
        print("  Features rebuilt")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(conn):
    print(f"\n{'='*50}")
    print("UPDATE COMPLETE — current DB state")
    print(f"{'='*50}")

    tables = [
        "matches", "statistics", "lineups", "standings",
        "xg", "odds", "features",
        "understat_player_stats", "understat_formation_stats",
    ]
    for t in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<30} {count:>6} rows")
        except Exception:
            print(f"  {t:<30} {'N/A':>6}")

    # Last match date
    last = conn.execute(
        "SELECT MAX(json_extract(raw_json, '$.fixture.date')) FROM matches"
    ).fetchone()[0]
    print(f"\n  Last match: {last}")

    # 2025/26 match count
    count_25 = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE season = '2025/26'"
    ).fetchone()[0]
    print(f"  2025/26 matches: {count_25}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Incremental data update")
    print(f"API key: {'*' * 4}{API_KEY[-8:] if API_KEY else 'NOT SET'}")
    print(f"Current season: {CURRENT_SEASON['label']}")

    # Override API key if set in environment, otherwise use the one from config
    if not API_KEY:
        print("\nERROR: API_FOOTBALL_KEY not set. Set it via environment variable.")
        print("  export API_FOOTBALL_KEY=your_key_here")
        sys.exit(1)

    conn = get_conn()
    init_db(conn)

    # Steps 1-3: api-sports.io (only new data)
    new_ids = update_fixtures(conn)
    update_statistics(conn, new_ids)
    update_lineups(conn, new_ids)
    update_standings(conn)

    # Steps 4-5: Understat (current season refresh)
    update_xg(conn)
    update_understat_advanced(conn)

    # Step 6: Odds CSV (current season)
    update_odds(conn)

    # Summary before features (features uses its own connection)
    print_summary(conn)
    conn.close()

    # Step 7: Rebuild features
    rebuild_features()

    print("\nDone.")
