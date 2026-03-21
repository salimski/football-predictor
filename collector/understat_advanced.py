"""
Advanced Understat scraper — player xG/xA stats and formation stats.

Endpoints:
  GET https://understat.com/getLeagueData/Bundesliga/{year}
    → data["players"] — list of player season stats (xG, xA, shots, etc.)

  GET https://understat.com/getTeamData/{team_slug}/{year}
    → data["statistics"]["formation"] — per-formation stats (xG for/against, games)

Stores results in:
  understat_player_stats  — player-level xG/xA per season
  understat_formation_stats — team formation effectiveness per season
"""

import json
import sqlite3
import time
import sys
import os

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import UNDERSTAT_SEASONS, DB_PATH, REQUEST_DELAY
from collector.normalize import normalize

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://understat.com/",
}

LEAGUE_URL = "https://understat.com/getLeagueData/Bundesliga/{year}"
TEAM_URL = "https://understat.com/getTeamData/{slug}/{year}"

LABEL_MAP = {
    "2021": "2021/22",
    "2022": "2022/23",
    "2023": "2023/24",
    "2024": "2024/25",
    "2025": "2025/26",
}

# Understat team name → URL slug mapping
# Understat uses English names; URL slugs replace spaces with underscores
UNDERSTAT_SLUG_MAP = {
    "Bayern Munich": "Bayern_Munich",
    "Borussia Dortmund": "Borussia_Dortmund",
    "Bayer Leverkusen": "Bayer_Leverkusen",
    "RasenBallsport Leipzig": "RasenBallsport_Leipzig",
    "Eintracht Frankfurt": "Eintracht_Frankfurt",
    "VfL Wolfsburg": "Wolfsburg",
    "Wolfsburg": "Wolfsburg",
    "Borussia M.Gladbach": "Borussia_M.Gladbach",
    "SC Freiburg": "Freiburg",
    "Freiburg": "Freiburg",
    "Hoffenheim": "Hoffenheim",
    "1899 Hoffenheim": "Hoffenheim",
    "Union Berlin": "Union_Berlin",
    "Mainz 05": "Mainz_05",
    "FSV Mainz 05": "Mainz_05",
    "Augsburg": "Augsburg",
    "FC Augsburg": "Augsburg",
    "VfB Stuttgart": "VfB_Stuttgart",
    "Werder Bremen": "Werder_Bremen",
    "FC Cologne": "FC_Cologne",
    "1. FC Köln": "FC_Cologne",
    "Bochum": "Bochum",
    "VfL Bochum": "Bochum",
    "Hertha Berlin": "Hertha_Berlin",
    "FC Schalke 04": "Schalke_04",
    "Schalke 04": "Schalke_04",
    "Arminia Bielefeld": "Arminia_Bielefeld",
    "Greuther Fuerth": "Greuther_Fuerth",
    "SpVgg Greuther Fürth": "Greuther_Fuerth",
    "Darmstadt": "Darmstadt",
    "SV Darmstadt 98": "Darmstadt",
    "FC Heidenheim": "FC_Heidenheim",
    "1. FC Heidenheim": "FC_Heidenheim",
    "Holstein Kiel": "Holstein_Kiel",
    "St. Pauli": "St._Pauli",
    "FC St. Pauli": "St._Pauli",
    "Hamburger SV": "Hamburger_SV",
    "Fortuna Düsseldorf": "Fortuna_Duesseldorf",
    "SV Elversberg": "SV_Elversberg",
}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Step 1: Player stats
# ---------------------------------------------------------------------------

def init_player_stats_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS understat_player_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            season          TEXT,
            player_id       TEXT,
            player_name     TEXT,
            team_name       TEXT,
            position        TEXT,
            games           INTEGER,
            minutes         INTEGER,
            goals           INTEGER,
            assists         INTEGER,
            xg              REAL,
            xa              REAL,
            xg_per90        REAL,
            xa_per90        REAL,
            shots           INTEGER,
            key_passes      INTEGER,
            goals_minus_xg  REAL,
            assists_minus_xa REAL,
            collected_at    TEXT DEFAULT (datetime('now')),
            UNIQUE(season, player_id)
        )
    """)
    conn.commit()


def collect_player_stats(conn):
    """Scrape player xG/xA stats from Understat for all 5 seasons."""
    init_player_stats_table(conn)

    total = 0
    for year in UNDERSTAT_SEASONS:
        label = LABEL_MAP[year]
        print(f"\n[understat players] Season {label} (year={year})")

        url = LEAGUE_URL.format(year=year)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [error] {exc}")
            time.sleep(REQUEST_DELAY)
            continue

        players = data.get("players", [])
        if not players:
            print("  No player data found")
            time.sleep(REQUEST_DELAY)
            continue

        inserted = 0
        skipped_minutes = 0
        for p in players:
            minutes = int(p.get("time", 0) or 0)
            if minutes < 90:
                skipped_minutes += 1
                continue

            player_id = str(p.get("id", ""))
            player_name = p.get("player_name", "")
            team_name = normalize(p.get("team_title", ""))
            position = p.get("position", "")
            games = int(p.get("games", 0) or 0)
            goals = int(p.get("goals", 0) or 0)
            assists = int(p.get("assists", 0) or 0)
            xg = float(p.get("xG", 0) or 0)
            xa = float(p.get("xA", 0) or 0)
            shots = int(p.get("shots", 0) or 0)
            key_passes = int(p.get("key_passes", 0) or 0)

            xg_per90 = (xg / minutes) * 90 if minutes > 0 else 0.0
            xa_per90 = (xa / minutes) * 90 if minutes > 0 else 0.0
            goals_minus_xg = goals - xg
            assists_minus_xa = assists - xa

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO understat_player_stats
                    (season, player_id, player_name, team_name, position,
                     games, minutes, goals, assists, xg, xa,
                     xg_per90, xa_per90, shots, key_passes,
                     goals_minus_xg, assists_minus_xa)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    label, player_id, player_name, team_name, position,
                    games, minutes, goals, assists, xg, xa,
                    xg_per90, xa_per90, shots, key_passes,
                    goals_minus_xg, assists_minus_xa,
                ))
                inserted += 1
            except Exception as exc:
                print(f"  [db error] {player_name}: {exc}")

        conn.commit()
        total += inserted
        print(f"  Stored {inserted} players (skipped {skipped_minutes} with <90 min)")

        # Sanity check: top 5 by xg_per90
        top5 = conn.execute("""
            SELECT player_name, team_name, xg_per90, xg, minutes, games
            FROM understat_player_stats
            WHERE season = ? AND minutes >= 450
            ORDER BY xg_per90 DESC LIMIT 5
        """, (label,)).fetchall()
        print(f"  Top 5 by xg_per90 (min 450 min):")
        for name, team, xgp90, xg_tot, mins, g in top5:
            print(f"    {name:25s} {team:25s} xG/90={xgp90:.3f} "
                  f"(xG={xg_tot:.1f}, {mins}min, {g}gm)")

        time.sleep(REQUEST_DELAY)

    print(f"\nTotal player rows inserted: {total}")
    return total


# ---------------------------------------------------------------------------
# Step 2: Formation stats
# ---------------------------------------------------------------------------

def init_formation_stats_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS understat_formation_stats (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            season              TEXT,
            team_name           TEXT,
            formation           TEXT,
            games               INTEGER,
            wins                INTEGER,
            draws               INTEGER,
            losses              INTEGER,
            goals_for           REAL,
            goals_against       REAL,
            xg_for              REAL,
            xg_against          REAL,
            xg_for_per_game     REAL,
            xg_against_per_game REAL,
            collected_at        TEXT DEFAULT (datetime('now')),
            UNIQUE(season, team_name, formation)
        )
    """)
    conn.commit()


def _get_teams_for_season(conn, year):
    """Get team names appearing in a season from the league data."""
    label = LABEL_MAP[year]
    url = LEAGUE_URL.format(year=year)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [error fetching league data] {exc}")
        return []

    teams_data = data.get("teams", {})
    teams = []
    for tid, tinfo in teams_data.items():
        title = tinfo.get("title", "")
        if title:
            teams.append(title)
    return sorted(teams)


def collect_formation_stats(conn):
    """Scrape formation stats from Understat team pages for all seasons."""
    init_formation_stats_table(conn)

    total = 0
    for year in UNDERSTAT_SEASONS:
        label = LABEL_MAP[year]
        print(f"\n[understat formations] Season {label} (year={year})")

        # Get teams for this season
        teams = _get_teams_for_season(conn, year)
        time.sleep(REQUEST_DELAY)

        if not teams:
            print("  No teams found")
            continue

        print(f"  Found {len(teams)} teams")
        season_inserted = 0

        for team_title in teams:
            slug = UNDERSTAT_SLUG_MAP.get(team_title)
            if not slug:
                # Try auto-generating slug
                slug = team_title.replace(" ", "_")
            canonical = normalize(team_title)

            url = TEAM_URL.format(slug=slug, year=year)
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"  [error] {team_title} ({slug}): {exc}")
                time.sleep(REQUEST_DELAY)
                continue

            time.sleep(REQUEST_DELAY)

            formations = data.get("statistics", {}).get("formation", {})
            if not formations:
                print(f"  [warn] No formation data for {team_title}")
                continue

            for form_name, fdata in formations.items():
                # Parse formation data
                total_time = int(fdata.get("time", 0) or 0)
                goals_for = int(fdata.get("goals", 0) or 0)
                xg_for = float(fdata.get("xG", 0) or 0)
                shots_for = int(fdata.get("shots", 0) or 0)

                against = fdata.get("against", {})
                goals_against = int(against.get("goals", 0) or 0)
                xg_against = float(against.get("xG", 0) or 0)

                # Estimate games from total_time (90 min per game)
                games = max(1, round(total_time / 90))

                # We don't have W/D/L directly from formation data,
                # so we estimate from the dates data
                wins = draws = losses = 0  # Will be computed from match results

                xg_for_pg = xg_for / games if games > 0 else 0.0
                xg_against_pg = xg_against / games if games > 0 else 0.0

                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO understat_formation_stats
                        (season, team_name, formation, games, wins, draws, losses,
                         goals_for, goals_against, xg_for, xg_against,
                         xg_for_per_game, xg_against_per_game)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        label, canonical, form_name, games, wins, draws, losses,
                        goals_for, goals_against, xg_for, xg_against,
                        xg_for_pg, xg_against_pg,
                    ))
                    season_inserted += 1
                except Exception as exc:
                    print(f"  [db error] {canonical} {form_name}: {exc}")

            conn.commit()

        total += season_inserted
        print(f"  Stored {season_inserted} formation rows")

    # Sanity check: Bayern Munich 2024/25
    print(f"\n--- Sanity check: Bayern Munich 2024/25 formations ---")
    rows = conn.execute("""
        SELECT formation, games, goals_for, goals_against,
               xg_for, xg_against, xg_for_per_game, xg_against_per_game
        FROM understat_formation_stats
        WHERE team_name = 'Bayern Munich' AND season = '2024/25'
        ORDER BY games DESC
    """).fetchall()
    for form, g, gf, ga, xgf, xga, xgfpg, xgapg in rows:
        print(f"  {form:12s}  {g:>2} games  GF={gf:.0f} GA={ga:.0f}  "
              f"xGF={xgf:.1f} xGA={xga:.1f}  "
              f"xGF/g={xgfpg:.2f} xGA/g={xgapg:.2f}")

    print(f"\nTotal formation rows inserted: {total}")
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    player_count = collect_player_stats(conn)
    formation_count = collect_formation_stats(conn)
    conn.close()

    print("\n" + "=" * 50)
    print("Understat advanced collection complete")
    print("=" * 50)
    print(f"  understat_player_stats:    {player_count} rows")
    print(f"  understat_formation_stats: {formation_count} rows")


if __name__ == "__main__":
    run()
