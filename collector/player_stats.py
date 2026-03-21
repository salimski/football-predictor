"""
Player statistics collector for api-sports.io.

Step 1: GET /players/squads?team={team_id}  — squad list per team
Step 2: GET /players?id={player_id}&season=2025 — individual player stats

Stores results in player_stats table in raw.db.
"""

import json
import sqlite3
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_KEY, API_HOST, BASE_URL, DB_PATH, REQUEST_DELAY
from collector.apifootball import api_get, get_conn


def init_player_stats_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            player_id     INTEGER,
            player_name   TEXT,
            team_id       INTEGER,
            team_name     TEXT,
            season        TEXT,
            position      TEXT,
            appearances   INTEGER,
            minutes       INTEGER,
            rating        REAL,
            goals         INTEGER,
            assists       INTEGER,
            shots_total   INTEGER,
            shots_on      INTEGER,
            passes_key    INTEGER,
            tackles_total INTEGER,
            interceptions INTEGER,
            duels_won     INTEGER,
            cards_yellow  INTEGER,
            cards_red     INTEGER,
            collected_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (player_id, team_id, season)
        )
    """)
    conn.commit()


def get_current_team_ids(conn):
    """Extract unique team IDs from 2025/26 matches."""
    rows = conn.execute("SELECT raw_json FROM matches WHERE season = '2025/26'").fetchall()
    team_ids = {}
    for (raw_json,) in rows:
        d = json.loads(raw_json)
        teams = d.get("teams", {})
        for side in ("home", "away"):
            t = teams.get(side, {})
            tid = t.get("id")
            tname = t.get("name", "")
            if tid:
                team_ids[tid] = tname
    return team_ids


def collect_squads(conn):
    """Step 1: Pull squad lists for all 18 Bundesliga teams.
    Returns {team_id: [(player_id, player_name), ...]}
    """
    team_ids = get_current_team_ids(conn)
    print(f"\nFound {len(team_ids)} teams in 2025/26 season")

    all_squads = {}
    for team_id, team_name in sorted(team_ids.items(), key=lambda x: x[1]):
        data = api_get("/players/squads", {"team": team_id})
        time.sleep(REQUEST_DELAY)

        if not data:
            print(f"  [WARN] No squad data for {team_name} (ID={team_id})")
            continue

        players = []
        # Response is a list with one element containing "players" array
        for entry in data:
            for p in entry.get("players", []):
                pid = p.get("id")
                pname = p.get("name", "")
                if pid:
                    players.append((pid, pname))

        all_squads[team_id] = players
        print(f"  {team_name:30s} (ID={team_id:>3d}): {len(players)} players")

    total = sum(len(v) for v in all_squads.values())
    print(f"\nTotal players across all squads: {total}")
    return all_squads, team_ids


def collect_player_stats(conn, all_squads, team_ids, season_year="2025"):
    """Step 2: Pull individual player stats and store in player_stats table.

    Uses GET /players?id={player_id}&season={season_year}
    Paginates if needed (API returns max 20 per page).
    """
    init_player_stats_table(conn)

    total_players = sum(len(v) for v in all_squads.values())
    print(f"\nCollecting stats for {total_players} players (season {season_year})...")

    inserted = 0
    null_ratings = 0
    counter = 0

    for team_id, players in all_squads.items():
        team_name = team_ids.get(team_id, f"Team {team_id}")

        for player_id, player_name in players:
            counter += 1
            if counter % 50 == 0:
                print(f"  [{counter}/{total_players}] Processing...")

            data = api_get("/players", {"id": player_id, "season": season_year})
            time.sleep(REQUEST_DELAY)

            if not data or len(data) == 0:
                continue

            # Each response element has "player" and "statistics" array
            player_info = data[0].get("player", {})
            stats_list = data[0].get("statistics", [])

            # Find the statistics entry for our league (Bundesliga = league 78)
            # or fallback to first entry
            stat = None
            for s in stats_list:
                league = s.get("league", {})
                if league.get("id") == 78:
                    stat = s
                    break
            if stat is None and stats_list:
                # Fallback: use first stats entry (might be another league)
                stat = stats_list[0]
            if stat is None:
                continue

            games = stat.get("games", {})
            goals_data = stat.get("goals", {})
            shots = stat.get("shots", {})
            passes = stat.get("passes", {})
            tackles = stat.get("tackles", {})
            duels = stat.get("duels", {})
            cards = stat.get("cards", {})

            rating_str = games.get("rating")
            rating = float(rating_str) if rating_str else None
            if rating is None:
                null_ratings += 1

            position = games.get("position") or player_info.get("position")

            row = (
                player_id,
                player_info.get("name", player_name),
                team_id,
                team_name,
                f"20{season_year[2:]}/{'26' if season_year == '2025' else str(int(season_year[2:]) + 1)}",
                position,
                games.get("appearences"),  # API typo: "appearences"
                games.get("minutes"),
                rating,
                goals_data.get("total"),
                goals_data.get("assists") or stat.get("goals", {}).get("assists"),
                shots.get("total"),
                shots.get("on"),
                passes.get("key"),
                tackles.get("total"),
                tackles.get("interceptions") or stat.get("tackles", {}).get("interceptions"),
                duels.get("won"),
                cards.get("yellow"),
                cards.get("red"),
            )

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO player_stats
                    (player_id, player_name, team_id, team_name, season,
                     position, appearances, minutes, rating, goals, assists,
                     shots_total, shots_on, passes_key, tackles_total,
                     interceptions, duels_won, cards_yellow, cards_red)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)
                inserted += 1
            except Exception as exc:
                print(f"  [db error] player {player_id}: {exc}")

        conn.commit()
        print(f"  {team_name}: committed")

    conn.commit()
    print(f"\nTotal rows inserted: {inserted}")
    print(f"Players with NULL rating: {null_ratings}")
    return inserted, null_ratings


def run():
    """Main entry point: collect squads then player stats."""
    conn = get_conn()
    all_squads, team_ids = collect_squads(conn)
    inserted, null_ratings = collect_player_stats(conn, all_squads, team_ids)
    conn.close()
    return inserted, null_ratings


if __name__ == "__main__":
    run()
