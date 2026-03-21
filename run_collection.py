"""
Phase 1 — Data Collection entry point.

Usage:
  python run_collection.py            # full 5-season pull
  python run_collection.py --dry-run  # one week of data to verify DB wiring
"""

import argparse
import json
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from config import DB_PATH, SEASONS
from collector import apifootball, understat as understat_collector


DRY_RUN_SEASON = {
    "label":   "dry-run",
    "from":    "2024-08-23",
    "to":      "2024-08-25",
    "dry_run": True,          # tells collect_events to pass from/to to the API
}

# ---------------------------------------------------------------------------
# Request budget estimate
# ---------------------------------------------------------------------------

def print_request_estimate():
    n              = len(SEASONS)
    matches_per_season = 380        # La Liga: 20 teams × 38 matchdays
    fixtures_calls  = n             # 1 per season
    standings_calls = n             # 1 per season
    stats_calls     = n * matches_per_season
    lineups_calls   = n * matches_per_season
    total_api       = fixtures_calls + standings_calls + stats_calls + lineups_calls

    print()
    print("=" * 50)
    print("REQUEST BUDGET ESTIMATE — full 5-season run")
    print("=" * 50)
    print(f"  GET /fixtures calls              {fixtures_calls:>5}  (1 per season)")
    print(f"  GET /standings calls             {standings_calls:>5}  (1 per season)")
    print(f"  GET /fixtures/statistics calls   {stats_calls:>5}  ({n} seasons × ~{matches_per_season} matches)")
    print(f"  GET /fixtures/lineups calls      {lineups_calls:>5}  ({n} seasons × ~{matches_per_season} matches)")
    print(f"  -----------------------------------------------")
    print(f"  Total api-sports.io              {total_api:>5}  requests")
    print()
    print("  Pro plan limit: 7,500 requests/day.")
    print(f"  Full pull needs ~{total_api} — spread across {total_api // 7500 + 1} days")
    print(f"  or run seasons one at a time (~{(fixtures_calls + standings_calls + stats_calls + lineups_calls) // n} requests/season).")
    print()
    print("  Understat scrapes the website (not API quota).")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n" + "=" * 50)
    print("FINAL SUMMARY — rows per table")
    print("=" * 50)
    conn = sqlite3.connect(DB_PATH)
    tables = ["matches", "statistics", "lineups", "standings", "xg"]
    for t in tables:
        try:
            (count,) = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
        except Exception:
            count = "table missing"
        print(f"  {t:<15} {str(count):>8}")
    conn.close()
    print("=" * 50)


# ---------------------------------------------------------------------------
# Dry-run sample printer
# ---------------------------------------------------------------------------

def print_sample_match():
    """Print the first match row from the DB in readable form."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT match_id, season, raw_json FROM matches LIMIT 1").fetchone()
    conn.close()

    if not row:
        print("\n  [sample] No match rows found.")
        return

    match_id, season, raw = row
    fx = json.loads(raw)   # one fixture object from data["response"]

    print("\n" + "=" * 50)
    print(f"SAMPLE MATCH  (match_id={match_id}, season={season})")
    print("=" * 50)

    # api-sports.io fixture structure: nested objects
    f  = fx.get("fixture", {})
    lg = fx.get("league",  {})
    tm = fx.get("teams",   {})
    gl = fx.get("goals",   {})

    fields = [
        ("fixture.id",          f.get("id")),
        ("fixture.date",        f.get("date")),
        ("fixture.status",      f.get("status", {}).get("long")),
        ("fixture.venue",       f.get("venue",  {}).get("name")),
        ("league.name",         lg.get("name")),
        ("league.round",        lg.get("round")),
        ("teams.home.name",     tm.get("home", {}).get("name")),
        ("teams.away.name",     tm.get("away", {}).get("name")),
        ("goals.home",          gl.get("home")),
        ("goals.away",          gl.get("away")),
    ]
    for label, value in fields:
        print(f"  {label:<30} {value}")

    # Check the separate statistics / lineups tables for this match
    conn2 = sqlite3.connect(DB_PATH)
    has_stats   = conn2.execute("SELECT 1 FROM statistics WHERE match_id = ?", (match_id,)).fetchone() is not None
    has_lineups = conn2.execute("SELECT 1 FROM lineups    WHERE match_id = ?", (match_id,)).fetchone() is not None
    conn2.close()

    print(f"  {'statistics stored':<30} {has_stats}")
    print(f"  {'lineups stored':<30} {has_lineups}")

    # Top-level keys present in the stored fixture object
    print(f"\n  Top-level keys: {list(fx.keys())}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football predictor — Phase 1 data collection")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull 2024-08-15 → 2024-08-25 (La Liga opener) to verify DB wiring, then print a sample row.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 50)
        print("DRY RUN — 2024-08-15 to 2024-08-25  (La Liga season opener)")
        print("Understat skipped in dry-run mode.")
        print("=" * 50)
        apifootball.run(seasons=[DRY_RUN_SEASON])
        print_sample_match()
        print_request_estimate()

    else:
        print("Phase 1: Data Collection — full 5-season run")
        print_request_estimate()

        print("\nRunning API-Football collector...")
        apifootball.run()

        print("\nRunning Understat xG scraper...")
        understat_collector.run()

        print_summary()
