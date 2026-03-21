"""
api-sports.io collector  (v3.football.api-sports.io)
Auth : headers  x-rapidapi-key / x-rapidapi-host  (NOT query params)

Endpoints used:
  GET /fixtures                       — finished matches per season
  GET /fixtures/statistics?fixture=ID — match statistics per fixture
  GET /fixtures/lineups?fixture=ID    — lineups per fixture
  GET /standings                      — league table per season

Stores raw JSON — no transformations.
"""

import json
import sqlite3
import time
import requests
import sys
import os
from urllib.parse import urlencode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_KEY, API_HOST, BASE_URL, LEAGUE_ID, SEASONS, DB_PATH, REQUEST_DELAY

HEADERS = {
    "x-rapidapi-key":  API_KEY,
    "x-rapidapi-host": API_HOST,
}


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id     TEXT PRIMARY KEY,
            season       TEXT,
            raw_json     TEXT,
            collected_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS statistics (
            match_id     TEXT PRIMARY KEY,
            season       TEXT,
            raw_json     TEXT,
            collected_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS lineups (
            match_id     TEXT PRIMARY KEY,
            season       TEXT,
            raw_json     TEXT,
            collected_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS standings (
            season       TEXT PRIMARY KEY,
            raw_json     TEXT,
            collected_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict, verbose: bool = False) -> list | None:
    """
    GET {BASE_URL}/{path}?{params} with auth headers.
    Returns data["response"] (a list) on success, None on any error.
    When verbose=True, prints the full raw response before any processing.
    """
    url = BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    print(f"  [GET] {url}?{urlencode(params)}")
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if verbose:
            print(f"  [raw response] {json.dumps(data, indent=2)}")

        errors = data.get("errors", {})
        if errors:
            print(f"  [API error] {errors}")
            return None

        return data.get("response")

    except Exception as exc:
        print(f"  [request failed] {exc}")
        return None


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_events(conn, season: dict) -> int:
    """
    Fetch finished fixtures for one season via GET /fixtures.
    Stores each raw fixture object in the matches table.
    Statistics and lineups are fetched separately — see collect_statistics /
    collect_lineups.
    Returns number of match rows stored.
    """
    label = season["label"]
    year  = season["from"][:4]   # "2024-08-15" → "2024"

    dry_run = season.get("dry_run", False)

    params: dict = {"league": LEAGUE_ID, "season": year}
    if dry_run:
        params["from"] = season["from"]
        params["to"]   = season["to"]
        print(f"\n[fixtures] {label}  year={year}  {season['from']} → {season['to']}  (no status filter)")
    else:
        params["status"] = "FT"
        print(f"\n[fixtures] Season {label}  year={year}")

    data = api_get("/fixtures", params, verbose=dry_run)
    time.sleep(REQUEST_DELAY)

    if data is None:
        print(f"  No data returned for {label}")
        return 0

    inserted = 0
    for fixture in data:
        match_id = str(fixture.get("fixture", {}).get("id", ""))
        if not match_id:
            continue
        try:
            conn.execute(
                "INSERT OR REPLACE INTO matches (match_id, season, raw_json) VALUES (?, ?, ?)",
                (match_id, label, json.dumps(fixture)),
            )
            inserted += 1
        except Exception as exc:
            print(f"  [db error] match {match_id}: {exc}")

    conn.commit()
    print(f"  Stored {inserted} matches")
    return inserted


def collect_statistics(conn, season: dict) -> int:
    """
    For every match in the season, call GET /fixtures/statistics?fixture=ID
    and store the response in the statistics table.
    Returns number of rows stored.
    """
    label = season["label"]
    print(f"\n[statistics] Season {label}")

    rows = conn.execute(
        "SELECT match_id FROM matches WHERE season = ?", (label,)
    ).fetchall()
    match_ids = [r[0] for r in rows]

    if not match_ids:
        print("  No matches found — run collect_events first")
        return 0

    inserted = 0
    for i, match_id in enumerate(match_ids, 1):
        print(f"  [{i}/{len(match_ids)}] fixture {match_id}", end="\r")
        data = api_get("/fixtures/statistics", {"fixture": match_id})
        time.sleep(REQUEST_DELAY)

        if not data:
            continue

        try:
            conn.execute(
                "INSERT OR REPLACE INTO statistics (match_id, season, raw_json) VALUES (?, ?, ?)",
                (match_id, label, json.dumps(data)),
            )
            inserted += 1
        except Exception as exc:
            print(f"\n  [db error] statistics {match_id}: {exc}")

    conn.commit()
    print(f"\n  Stored {inserted} statistics rows")
    return inserted


def collect_lineups(conn, season: dict) -> int:
    """
    For every match in the season, call GET /fixtures/lineups?fixture=ID
    and store the response in the lineups table.
    Returns number of rows stored.
    """
    label = season["label"]
    print(f"\n[lineups] Season {label}")

    rows = conn.execute(
        "SELECT match_id FROM matches WHERE season = ?", (label,)
    ).fetchall()
    match_ids = [r[0] for r in rows]

    if not match_ids:
        print("  No matches found — run collect_events first")
        return 0

    inserted = 0
    for i, match_id in enumerate(match_ids, 1):
        print(f"  [{i}/{len(match_ids)}] fixture {match_id}", end="\r")
        data = api_get("/fixtures/lineups", {"fixture": match_id})
        time.sleep(REQUEST_DELAY)

        if not data:
            continue

        try:
            conn.execute(
                "INSERT OR REPLACE INTO lineups (match_id, season, raw_json) VALUES (?, ?, ?)",
                (match_id, label, json.dumps(data)),
            )
            inserted += 1
        except Exception as exc:
            print(f"\n  [db error] lineups {match_id}: {exc}")

    conn.commit()
    print(f"\n  Stored {inserted} lineups rows")
    return inserted


def collect_standings(conn, season: dict) -> int:
    """Fetch standings for one season via GET /standings. Returns 1 on success."""
    label = season["label"]
    year  = season["from"][:4]
    print(f"\n[standings] Season {label}  year={year}")

    data = api_get("/standings", {"league": LEAGUE_ID, "season": year})
    time.sleep(REQUEST_DELAY)

    if not data:
        print(f"  No standings data for {label}")
        return 0

    try:
        conn.execute(
            "INSERT OR REPLACE INTO standings (season, raw_json) VALUES (?, ?)",
            (label, json.dumps(data)),
        )
        conn.commit()
        print(f"  Stored standings for {label}")
        return 1
    except Exception as exc:
        print(f"  [db error] standings {label}: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(seasons: list[dict] | None = None) -> dict:
    """
    Collect all data for the given seasons (defaults to all 5 in config).
    Order per season: fixtures → statistics → lineups → standings.
    Returns totals dict for use by the caller.
    """
    if seasons is None:
        seasons = SEASONS

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    init_db(conn)

    totals = {"matches": 0, "statistics": 0, "lineups": 0, "standings": 0}

    for season in seasons:
        totals["matches"]    += collect_events(conn, season)
        totals["statistics"] += collect_statistics(conn, season)
        totals["lineups"]    += collect_lineups(conn, season)
        totals["standings"]  += collect_standings(conn, season)

    conn.close()

    print("\n" + "=" * 50)
    print("api-sports.io collection complete")
    print("=" * 50)
    for table, count in totals.items():
        print(f"  {table:<15} {count:>6} rows")

    return totals


if __name__ == "__main__":
    run()
