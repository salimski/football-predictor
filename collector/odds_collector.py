"""
Collect over/under 2.5 odds from football-data.co.uk CSV files.

Reads Bundesliga CSVs (D1 division only), normalizes team names,
converts odds to implied probabilities, and writes to the `odds` table.

Usage:
    python -m collector.odds_collector
"""

import os
import sys
import sqlite3
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH
from collector.normalize import normalize

CSV_DIR = os.path.join("data", "external", "football_data_co_uk")

# Map filename prefix to season label (matching config.SEASONS)
SEASON_MAP = {
    "21_22": "2021/22",
    "22_23": "2022/23",
    "23_24": "2023/24",
    "24_25": "2024/25",
    "25_26": "2025/26",
}

# Columns to extract (football-data.co.uk naming)
ODDS_COLS = {
    "B365>2.5":  "raw_b365_over",
    "B365<2.5":  "raw_b365_under",
    "Avg>2.5":   "raw_bb_avg_over",
    "Avg<2.5":   "raw_bb_avg_under",
    "Max>2.5":   "raw_bb_max_over",
    "Max<2.5":   "raw_bb_max_under",
}


def parse_date(date_str):
    """Convert DD/MM/YYYY or DD/MM/YY to YYYY-MM-DD."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def odds_to_prob(odds):
    """Convert decimal odds to implied probability."""
    if pd.isna(odds) or odds <= 0:
        return None
    return 1.0 / odds


def load_csv(filepath, season):
    """Load a single CSV and return cleaned DataFrame."""
    df = pd.read_csv(filepath)

    # Only Bundesliga 1 (D1) — skip if Div column exists and isn't D1
    if "Div" in df.columns:
        df = df[df["Div"] == "D1"]

    if len(df) == 0:
        return pd.DataFrame()

    records = []
    now = datetime.now().isoformat()

    for _, row in df.iterrows():
        date_str = parse_date(str(row.get("Date", "")))
        if date_str is None:
            continue

        home = normalize(str(row.get("HomeTeam", "")))
        away = normalize(str(row.get("AwayTeam", "")))

        raw_b365_over  = row.get("B365>2.5")
        raw_b365_under = row.get("B365<2.5")
        raw_avg_over   = row.get("Avg>2.5")
        raw_avg_under  = row.get("Avg<2.5")
        raw_max_over   = row.get("Max>2.5")
        raw_pin_over   = row.get("P>2.5")
        raw_pin_under  = row.get("P<2.5")

        records.append({
            "season":              season,
            "match_date":          date_str,
            "home_team":           home,
            "away_team":           away,
            "b365_prob_over25":    odds_to_prob(raw_b365_over),
            "b365_prob_under25":   odds_to_prob(raw_b365_under),
            "bb_avg_prob_over25":  odds_to_prob(raw_avg_over),
            "bb_avg_prob_under25": odds_to_prob(raw_avg_under),
            "bb_max_prob_over25":  odds_to_prob(raw_max_over),
            "raw_b365_over":       raw_b365_over if pd.notna(raw_b365_over) else None,
            "raw_b365_under":      raw_b365_under if pd.notna(raw_b365_under) else None,
            "pinnacle_prob_over25":  odds_to_prob(raw_pin_over),
            "pinnacle_prob_under25": odds_to_prob(raw_pin_under),
            "collected_at":        now,
        })

    return pd.DataFrame(records)


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT,
            match_date TEXT,
            home_team TEXT,
            away_team TEXT,
            b365_prob_over25 REAL,
            b365_prob_under25 REAL,
            bb_avg_prob_over25 REAL,
            bb_avg_prob_under25 REAL,
            bb_max_prob_over25 REAL,
            raw_b365_over REAL,
            raw_b365_under REAL,
            pinnacle_prob_over25 REAL,
            pinnacle_prob_under25 REAL,
            collected_at TEXT
        )
    """)
    conn.commit()


def run():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # Drop and recreate for clean import
    conn.execute("DROP TABLE IF EXISTS odds")
    create_table(conn)

    total_inserted = 0

    for prefix, season in sorted(SEASON_MAP.items()):
        # Only use (1) files — Bundesliga 1
        filename = f"{prefix}(1).csv"
        filepath = os.path.join(CSV_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  SKIP {filename} — not found")
            continue

        df = load_csv(filepath, season)
        if len(df) == 0:
            print(f"  SKIP {filename} — no D1 rows")
            continue

        # Insert
        df.to_sql("odds", conn, if_exists="append", index=False)
        print(f"  {season}: {len(df)} rows inserted from {filename}")
        total_inserted += len(df)

    conn.commit()
    print(f"\nTotal: {total_inserted} rows inserted into odds table")

    # Match quality report
    print("\n--- Match Report ---")
    for prefix, season in sorted(SEASON_MAP.items()):
        odds_count = conn.execute(
            "SELECT COUNT(*) FROM odds WHERE season = ?", (season,)
        ).fetchone()[0]

        matched = conn.execute("""
            SELECT COUNT(*) FROM odds o
            JOIN matches m
              ON o.match_date = substr(json_extract(m.raw_json, '$.fixture.date'), 1, 10)
             AND o.home_team = m.home_team
             AND o.away_team = m.away_team
            WHERE o.season = ?
        """, (season,)).fetchone()[0]

        # Check for NULL odds
        null_b365 = conn.execute(
            "SELECT COUNT(*) FROM odds WHERE season = ? AND b365_prob_over25 IS NULL",
            (season,)
        ).fetchone()[0]

        null_pin = conn.execute(
            "SELECT COUNT(*) FROM odds WHERE season = ? AND pinnacle_prob_over25 IS NULL",
            (season,)
        ).fetchone()[0]

        print(f"  {season}: {odds_count} odds rows, {matched} matched to matches"
              f" ({matched}/{odds_count}), {null_b365} NULL B365, {null_pin} NULL Pinnacle")

    # Total match
    total_matched = conn.execute("""
        SELECT COUNT(*) FROM odds o
        JOIN matches m
          ON o.match_date = substr(json_extract(m.raw_json, '$.fixture.date'), 1, 10)
         AND o.home_team = m.home_team
         AND o.away_team = m.away_team
    """).fetchone()[0]
    print(f"\n  Total matched: {total_matched} / {total_inserted}")

    # Show unmatched for debugging
    unmatched = conn.execute("""
        SELECT o.season, o.match_date, o.home_team, o.away_team
        FROM odds o
        LEFT JOIN matches m
          ON o.match_date = substr(json_extract(m.raw_json, '$.fixture.date'), 1, 10)
         AND o.home_team = m.home_team
         AND o.away_team = m.away_team
        WHERE m.match_id IS NULL
        LIMIT 20
    """).fetchall()
    if unmatched:
        print(f"\n  First {len(unmatched)} unmatched odds rows:")
        for s, d, h, a in unmatched:
            print(f"    {s}  {d}  {h} vs {a}")

    conn.close()


if __name__ == "__main__":
    run()
