"""
Understat xG scraper for Bundesliga.

Understat serves match data via a JSON API endpoint:
    GET https://understat.com/getLeagueData/{league}/{year}
Returns {"dates": [...], "teams": {...}, "players": {...}}
Match data is in data["dates"] — same field structure as before.

Compatible with Python 3.14+.
"""

import json
import sqlite3
import time
import sys
import os

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import UNDERSTAT_SEASONS, DB_PATH, REQUEST_DELAY

UNDERSTAT_URL = "https://understat.com/getLeagueData/Bundesliga/{year}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_xg_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS xg (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            season       TEXT,
            match_date   TEXT,
            home_team    TEXT,
            away_team    TEXT,
            home_xg      REAL,
            away_xg      REAL,
            home_goals   INTEGER,
            away_goals   INTEGER,
            understat_id TEXT UNIQUE,
            raw_json     TEXT,
            collected_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def fetch_season_xg(year: str) -> list[dict]:
    """
    Fetch Bundesliga match results with xG for a season.
    `year` is the starting year, e.g. "2021" for the 2021/22 season.
    Returns the list of match dicts from data["dates"].
    """
    url = UNDERSTAT_URL.format(year=year)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    matches = data.get("dates")
    if not matches:
        raise ValueError(f"No 'dates' key in response for season {year}")
    return matches


def store_xg(conn, season_label: str, matches: list[dict]) -> int:
    inserted = 0
    for m in matches:
        understat_id = str(m.get("id", ""))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO xg
                    (season, match_date, home_team, away_team,
                     home_xg, away_xg, home_goals, away_goals,
                     understat_id, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    season_label,
                    m.get("datetime", "")[:10],        # YYYY-MM-DD
                    m.get("h", {}).get("title", ""),
                    m.get("a", {}).get("title", ""),
                    float(m.get("xG",    {}).get("h", 0) or 0),
                    float(m.get("xG",    {}).get("a", 0) or 0),
                    int(m.get("goals", {}).get("h", 0) or 0),
                    int(m.get("goals", {}).get("a", 0) or 0),
                    understat_id,
                    json.dumps(m),
                ),
            )
            inserted += 1
        except Exception as exc:
            print(f"  [db error] match {understat_id}: {exc}")

    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "2021": "2021/22",
    "2022": "2022/23",
    "2023": "2023/24",
    "2024": "2024/25",
    "2025": "2025/26",
}


def run():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    init_xg_table(conn)

    total = 0
    for year in UNDERSTAT_SEASONS:
        label = LABEL_MAP.get(year, year)
        print(f"\n[xG] Fetching Bundesliga {label} from Understat...", end=" ", flush=True)
        try:
            matches = fetch_season_xg(year)
            count = store_xg(conn, label, matches)
            print(f"{count} matches stored")
            total += count
        except Exception as exc:
            print(f"\n  [error] Season {label}: {exc}")
        time.sleep(REQUEST_DELAY)

    conn.close()

    print("\n" + "=" * 50)
    print("Understat xG collection complete")
    print("=" * 50)
    print(f"  {'xg':<15} {total:>6} rows")


if __name__ == "__main__":
    run()
