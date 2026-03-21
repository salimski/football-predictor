"""
Player-based team strength features.

Computes per match:
  h_squad_avg_rating      — avg rating of home team's starting XI
  a_squad_avg_rating      — avg rating of away team's starting XI
  h_squad_depth_rating    — minutes-weighted avg rating of full squad
  a_squad_depth_rating    — minutes-weighted avg rating of full squad

Uses the lineups table (starting XI player IDs) joined to player_stats
(season ratings). Falls back to tier-based rating when data is missing.
"""

import sqlite3
import json
import sys
import os
import unicodedata
from difflib import get_close_matches

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH
from collector.normalize import normalize


# ---------------------------------------------------------------------------
# Tier-based fallback rating
# ---------------------------------------------------------------------------

def build_tier_rating_map(conn):
    """Build {team_name: fallback_rating} from standings.

    Maps standings rank (1-18) to a 6.5-8.0 scale linearly.
    Rank 1 -> 8.0, Rank 18 -> 6.5.
    Teams not found get 7.0 (midpoint).
    """
    rows = conn.execute("SELECT season, raw_json FROM standings ORDER BY season DESC").fetchall()
    if not rows:
        return {}

    # Use the most recent completed season for fallback
    # Try 2024/25 first, then latest available
    tier_map = {}
    for season, raw_json in rows:
        d = json.loads(raw_json)
        try:
            table = d[0]["league"]["standings"][0]
        except (KeyError, IndexError, TypeError):
            continue
        for entry in table:
            team_name = normalize(entry.get("team", {}).get("name", ""))
            rank = entry.get("rank", 10)
            # Linear map: rank 1 -> 8.0, rank 18 -> 6.5
            rating = 8.0 - (rank - 1) * (1.5 / 17)
            if team_name not in tier_map:  # Prefer most recent season
                tier_map[team_name] = rating
        break  # Use only the most recent season

    return tier_map


# ---------------------------------------------------------------------------
# Player stats lookup
# ---------------------------------------------------------------------------

def load_player_stats(conn):
    """Load player_stats table into a DataFrame."""
    try:
        df = pd.read_sql("SELECT * FROM player_stats", conn)
        return df
    except Exception:
        return pd.DataFrame()


def build_player_rating_lookup(player_stats_df):
    """Build {player_id: rating} and {(team_id, position): avg_rating} lookups."""
    player_ratings = {}
    team_pos_ratings = {}

    if player_stats_df.empty:
        return player_ratings, team_pos_ratings

    for _, row in player_stats_df.iterrows():
        pid = row["player_id"]
        rating = row["rating"]
        if pd.notna(rating) and rating > 0:
            player_ratings[pid] = rating

    # Team-position average ratings (for fallback when specific player not found)
    grouped = player_stats_df[player_stats_df["rating"].notna()].groupby(
        ["team_id", "position"]
    )["rating"].mean()
    for (team_id, pos), avg in grouped.items():
        team_pos_ratings[(team_id, pos)] = avg

    return player_ratings, team_pos_ratings


def build_team_depth_ratings(player_stats_df):
    """Build {team_id: minutes-weighted average rating} for full squad depth.

    Players with more minutes get higher weight.
    """
    if player_stats_df.empty:
        return {}

    depth = {}
    for team_id, group in player_stats_df.groupby("team_id"):
        valid = group[group["rating"].notna() & group["minutes"].notna()]
        valid = valid[valid["minutes"] > 0]
        if valid.empty:
            continue
        weights = valid["minutes"].values.astype(float)
        ratings = valid["rating"].values.astype(float)
        depth[team_id] = float(np.average(ratings, weights=weights))

    return depth


# ---------------------------------------------------------------------------
# Extract starting XI player IDs from lineups
# ---------------------------------------------------------------------------

def extract_starting_xi_ids(lineup_json, side_idx):
    """Extract player IDs from startXI of a lineup.

    Args:
        lineup_json: parsed JSON from lineups table raw_json
        side_idx: 0 for home, 1 for away

    Returns:
        list of (player_id, position) tuples
    """
    try:
        side = lineup_json[side_idx]
        team_id = side.get("team", {}).get("id")
        start_xi = side.get("startXI", [])
        players = []
        for entry in start_xi:
            p = entry.get("player", {})
            pid = p.get("id")
            pos = p.get("pos")  # G, D, M, F
            if pid:
                players.append((pid, pos))
        return players, team_id
    except (IndexError, KeyError, TypeError):
        return [], None


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_squad_features(db_path=DB_PATH):
    """Compute squad rating features for all matches.

    Returns a DataFrame with columns:
        match_id, h_squad_avg_rating, a_squad_avg_rating,
        h_squad_depth_rating, a_squad_depth_rating
    """
    conn = sqlite3.connect(db_path)

    # Load data
    player_stats_df = load_player_stats(conn)
    if player_stats_df.empty:
        print("WARNING: player_stats table is empty. Run collector/player_stats.py first.")
        conn.close()
        return pd.DataFrame(columns=[
            "match_id", "h_squad_avg_rating", "a_squad_avg_rating",
            "h_squad_depth_rating", "a_squad_depth_rating",
        ])

    player_ratings, team_pos_ratings = build_player_rating_lookup(player_stats_df)
    depth_ratings = build_team_depth_ratings(player_stats_df)
    tier_map = build_tier_rating_map(conn)

    # Load lineups
    lineups = conn.execute("SELECT match_id, raw_json FROM lineups").fetchall()

    # Also need match -> team mapping for tier fallback
    matches = conn.execute(
        "SELECT match_id, raw_json FROM matches"
    ).fetchall()
    match_teams = {}
    for mid, rj in matches:
        d = json.loads(rj)
        teams = d.get("teams", {})
        h = teams.get("home", {})
        a = teams.get("away", {})
        match_teams[mid] = {
            "home_id": h.get("id"),
            "home_name": normalize(h.get("name", "")),
            "away_id": a.get("id"),
            "away_name": normalize(a.get("name", "")),
        }

    conn.close()

    DEFAULT_RATING = 7.0

    records = []
    for match_id, raw_json in lineups:
        d = json.loads(raw_json)

        mt = match_teams.get(match_id, {})

        # Home team starting XI
        h_players, h_team_id = extract_starting_xi_ids(d, 0)
        h_team_name = mt.get("home_name", "")
        h_fallback = tier_map.get(h_team_name, DEFAULT_RATING)

        h_ratings = []
        for pid, pos in h_players:
            if pos == "G":
                continue  # Skip goalkeepers
            r = player_ratings.get(pid)
            if r is None:
                # Fallback: team-position average
                r = team_pos_ratings.get((h_team_id, _map_pos(pos)))
            if r is None:
                r = h_fallback
            h_ratings.append(r)

        # Away team starting XI
        a_players, a_team_id = extract_starting_xi_ids(d, 1)
        a_team_name = mt.get("away_name", "")
        a_fallback = tier_map.get(a_team_name, DEFAULT_RATING)

        a_ratings = []
        for pid, pos in a_players:
            if pos == "G":
                continue
            r = player_ratings.get(pid)
            if r is None:
                r = team_pos_ratings.get((a_team_id, _map_pos(pos)))
            if r is None:
                r = a_fallback
            a_ratings.append(r)

        h_avg = float(np.mean(h_ratings)) if h_ratings else h_fallback
        a_avg = float(np.mean(a_ratings)) if a_ratings else a_fallback

        h_depth = depth_ratings.get(h_team_id or mt.get("home_id"), h_fallback)
        a_depth = depth_ratings.get(a_team_id or mt.get("away_id"), a_fallback)

        records.append({
            "match_id": match_id,
            "h_squad_avg_rating": round(h_avg, 3),
            "a_squad_avg_rating": round(a_avg, 3),
            "h_squad_depth_rating": round(h_depth, 3),
            "a_squad_depth_rating": round(a_depth, 3),
        })

    df = pd.DataFrame(records)
    print(f"Computed squad features for {len(df)} matches")
    print(f"  h_squad_avg_rating   mean={df['h_squad_avg_rating'].mean():.3f}  "
          f"std={df['h_squad_avg_rating'].std():.3f}")
    print(f"  a_squad_avg_rating   mean={df['a_squad_avg_rating'].mean():.3f}  "
          f"std={df['a_squad_avg_rating'].std():.3f}")
    print(f"  h_squad_depth_rating mean={df['h_squad_depth_rating'].mean():.3f}  "
          f"std={df['h_squad_depth_rating'].std():.3f}")
    print(f"  a_squad_depth_rating mean={df['a_squad_depth_rating'].mean():.3f}  "
          f"std={df['a_squad_depth_rating'].std():.3f}")
    return df


def _map_pos(pos_code):
    """Map lineup pos code (G/D/M/F) to player_stats position name."""
    mapping = {
        "G": "Goalkeeper",
        "D": "Defender",
        "M": "Midfielder",
        "F": "Attacker",
    }
    return mapping.get(pos_code, pos_code)


# ---------------------------------------------------------------------------
# Data-driven injury adjustment
# ---------------------------------------------------------------------------

def get_team_player_ratings(team_name, db_path=DB_PATH):
    """Get player ratings for a team, grouped by position.

    Returns:
        {
            'Defender': [(player_name, rating), ...],
            'Midfielder': [(player_name, rating), ...],
            'Attacker': [(player_name, rating), ...],
            'avg_by_pos': {'Defender': avg, 'Midfielder': avg, 'Attacker': avg},
            'team_avg': overall_avg,
        }
    """
    conn = sqlite3.connect(db_path)

    # Get team_id from matches
    row = conn.execute("""
        SELECT raw_json FROM matches WHERE season = '2025/26' LIMIT 1
    """).fetchone()

    # Find team_id by name
    all_matches = conn.execute(
        "SELECT raw_json FROM matches WHERE season = '2025/26'"
    ).fetchall()
    team_id = None
    for (rj,) in all_matches:
        d = json.loads(rj)
        for side in ("home", "away"):
            t = d.get("teams", {}).get(side, {})
            if normalize(t.get("name", "")) == team_name:
                team_id = t.get("id")
                break
        if team_id:
            break

    if team_id is None:
        conn.close()
        return None

    players = conn.execute("""
        SELECT player_name, position, rating, minutes
        FROM player_stats
        WHERE team_id = ? AND rating IS NOT NULL
        ORDER BY position, rating DESC
    """, (team_id,)).fetchall()
    conn.close()

    result = {
        "Defender": [],
        "Midfielder": [],
        "Attacker": [],
        "avg_by_pos": {},
        "team_avg": 0.0,
    }

    all_ratings = []
    for name, pos, rating, minutes in players:
        if pos in result and pos != "Goalkeeper":
            result[pos].append((name, rating))
            all_ratings.append(rating)
        elif pos == "Goalkeeper":
            continue
        else:
            all_ratings.append(rating)

    for pos in ("Defender", "Midfielder", "Attacker"):
        ratings = [r for _, r in result[pos]]
        result["avg_by_pos"][pos] = float(np.mean(ratings)) if ratings else 7.0

    result["team_avg"] = float(np.mean(all_ratings)) if all_ratings else 7.0

    return result


INJURY_POSITION_WEIGHTS = {
    "Defender": 0.8,
    "Midfielder": 0.6,
    "Attacker": 0.7,
}


def data_driven_injury_adjustment(base_prob, injured_players, team_data):
    """Compute injury adjustment based on actual player ratings.

    Args:
        base_prob: blended probability before injury adjustment
        injured_players: list of dicts with keys:
            - 'name' (str, optional): player name for exact lookup
            - 'position' (str): 'Defender', 'Midfielder', or 'Attacker'
            - 'side' (str): 'home' or 'away'
        team_data: dict with 'home' and 'away' keys, each from get_team_player_ratings()

    Returns:
        (adjusted_prob, adjustment_details)
    """
    total_adj = 0.0
    details = []

    for inj in injured_players:
        pos = inj["position"]
        side = inj["side"]
        name = inj.get("name")

        td = team_data.get(side)
        if td is None:
            continue

        team_avg = td["team_avg"]
        pos_avg = td["avg_by_pos"].get(pos, team_avg)
        pos_weight = INJURY_POSITION_WEIGHTS.get(pos, 0.6)

        # Find specific player rating, or use position average
        player_rating = pos_avg  # default
        if name:
            for pname, prating in td.get(pos, []):
                if name.lower() in pname.lower() or pname.lower() in name.lower():
                    player_rating = prating
                    break

        # Adjustment = (player_rating - team_avg) * position_weight * direction
        # Higher rated player missing = bigger impact
        # Defender/midfielder missing -> push toward over (more goals)
        # Attacker missing -> push toward under (fewer goals)
        rating_diff = player_rating - team_avg
        if pos in ("Defender", "Midfielder"):
            direction = +1  # missing defender/mid -> more goals -> over
        else:
            direction = -1  # missing attacker -> fewer goals -> under

        # Base adjustment per missing player + extra for above-average players
        base_adj = 0.01 * direction  # 1% base per missing player
        quality_adj = rating_diff * pos_weight * 0.02 * direction  # quality premium
        adj = base_adj + quality_adj

        total_adj += adj
        details.append({
            "name": name or f"Unknown {pos}",
            "position": pos,
            "side": side,
            "rating": player_rating,
            "adjustment": adj,
        })

    adjusted = max(0.0, min(1.0, base_prob + total_adj))
    return adjusted, details


def apply_injury_adjustment_v2(base_prob, injuries, home_team, away_team, db_path=DB_PATH):
    """Drop-in replacement for predict.py's apply_injury_adjustment.

    Args:
        base_prob: blended probability
        injuries: dict with keys h_def, h_mid, h_atk, a_def, a_mid, a_atk (counts)
            OR list of dicts with 'name', 'position', 'side' keys
        home_team: home team name
        away_team: away team name

    Returns:
        (adjusted_prob, details)
    """
    # Load team data
    home_data = get_team_player_ratings(home_team, db_path)
    away_data = get_team_player_ratings(away_team, db_path)
    team_data = {"home": home_data, "away": away_data}

    # Convert count-based injuries to list format
    if isinstance(injuries, dict) and "h_def" in injuries:
        injured_players = []
        pos_map = {
            "h_def": ("Defender", "home"),
            "h_mid": ("Midfielder", "home"),
            "h_atk": ("Attacker", "home"),
            "a_def": ("Defender", "away"),
            "a_mid": ("Midfielder", "away"),
            "a_atk": ("Attacker", "away"),
        }
        for key, (pos, side) in pos_map.items():
            count = injuries.get(key, 0)
            for _ in range(count):
                injured_players.append({"position": pos, "side": side})
    else:
        injured_players = injuries

    return data_driven_injury_adjustment(base_prob, injured_players, team_data)


# ---------------------------------------------------------------------------
# Understat-based features: formation xG and top-player xG/xA
# ---------------------------------------------------------------------------

def _load_formation_lookup(conn):
    """Build {(season, team_name, formation): {xg_for_pg, xg_against_pg}}
    and {(season, team_name): most_used_formation} lookups.
    """
    try:
        rows = conn.execute("""
            SELECT season, team_name, formation, games,
                   xg_for_per_game, xg_against_per_game
            FROM understat_formation_stats
        """).fetchall()
    except Exception:
        return {}, {}

    formation_data = {}
    team_most_used = {}  # (season, team) -> (formation, games)

    for season, team, formation, games, xgf_pg, xga_pg in rows:
        key = (season, team, formation)
        formation_data[key] = {
            "xg_for_per_game": xgf_pg,
            "xg_against_per_game": xga_pg,
            "games": games,
        }
        # Track most-used formation per team-season
        tk = (season, team)
        if tk not in team_most_used or games > team_most_used[tk][1]:
            team_most_used[tk] = (formation, games)

    return formation_data, team_most_used


def _load_player_xg_lookup(conn):
    """Build {(season, team_name): [(player_name, xg_per90, xa_per90), ...]}
    sorted by xg_per90 descending.
    """
    try:
        rows = conn.execute("""
            SELECT season, team_name, player_name, xg_per90, xa_per90
            FROM understat_player_stats
            WHERE minutes >= 450
            ORDER BY season, team_name, xg_per90 DESC
        """).fetchall()
    except Exception:
        return {}

    lookup = {}
    for season, team, name, xgp90, xap90 in rows:
        key = (season, team)
        lookup.setdefault(key, []).append((name, xgp90, xap90))
    return lookup


def _get_season_label(match_season):
    """Get the current season label. For feature lookup, we use the match's own
    season — this ensures strict temporal ordering since Understat stats are
    cumulative within a season and we only look at current/prior seasons."""
    return match_season


def _prior_season_label(season):
    """Get the prior season label (e.g., '2024/25' -> '2023/24')."""
    from config import SEASONS
    labels = [s["label"] for s in SEASONS]
    try:
        idx = labels.index(season)
        return labels[idx - 1] if idx > 0 else None
    except ValueError:
        return None


def compute_understat_features(db_path=DB_PATH):
    """Compute Understat-based features for all matches.

    Returns DataFrame with columns:
        match_id, h_formation_xg_per_game, a_formation_xg_per_game,
        h_formation_xga_per_game, a_formation_xga_per_game,
        formation_xg_matchup,
        h_top3_xg_per90, a_top3_xg_per90,
        h_top3_xa_per90, a_top3_xa_per90
    """
    conn = sqlite3.connect(db_path)

    formation_data, team_most_used = _load_formation_lookup(conn)
    player_xg_lookup = _load_player_xg_lookup(conn)

    if not formation_data and not player_xg_lookup:
        print("WARNING: No Understat advanced data. Run collector/understat_advanced.py first.")
        conn.close()
        cols = ["match_id", "h_formation_xg_per_game", "a_formation_xg_per_game",
                "h_formation_xga_per_game", "a_formation_xga_per_game",
                "formation_xg_matchup",
                "h_top3_xg_per90", "a_top3_xg_per90",
                "h_top3_xa_per90", "a_top3_xa_per90"]
        return pd.DataFrame(columns=cols)

    # Load matches with formations
    matches = conn.execute("""
        SELECT m.match_id, m.season, m.home_team, m.away_team, l.raw_json
        FROM matches m
        LEFT JOIN lineups l ON m.match_id = l.match_id
    """).fetchall()
    conn.close()

    records = []
    for match_id, season, home_team, away_team, lineup_json in matches:
        # Parse formations from lineup
        h_formation = a_formation = None
        if lineup_json:
            try:
                ld = json.loads(lineup_json)
                h_formation = ld[0].get("formation")
                a_formation = ld[1].get("formation")
            except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                pass

        # --- Formation xG features ---
        h_fxg = _get_formation_xg(season, home_team, h_formation,
                                   formation_data, team_most_used)
        a_fxg = _get_formation_xg(season, away_team, a_formation,
                                   formation_data, team_most_used)

        h_formation_xg_pg = h_fxg["xg_for_per_game"] if h_fxg else None
        a_formation_xg_pg = a_fxg["xg_for_per_game"] if a_fxg else None
        h_formation_xga_pg = h_fxg["xg_against_per_game"] if h_fxg else None
        a_formation_xga_pg = a_fxg["xg_against_per_game"] if a_fxg else None

        formation_xg_matchup = None
        if h_formation_xg_pg is not None and a_formation_xg_pg is not None:
            formation_xg_matchup = h_formation_xg_pg + a_formation_xg_pg

        # --- Top 3 player xG/xA features ---
        h_top3 = _get_top3_xg(season, home_team, player_xg_lookup)
        a_top3 = _get_top3_xg(season, away_team, player_xg_lookup)

        records.append({
            "match_id": match_id,
            "h_formation_xg_per_game": h_formation_xg_pg,
            "a_formation_xg_per_game": a_formation_xg_pg,
            "h_formation_xga_per_game": h_formation_xga_pg,
            "a_formation_xga_per_game": a_formation_xga_pg,
            "formation_xg_matchup": formation_xg_matchup,
            "h_top3_xg_per90": h_top3[0],
            "a_top3_xg_per90": a_top3[0],
            "h_top3_xa_per90": h_top3[1],
            "a_top3_xa_per90": a_top3[1],
        })

    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c != "match_id"]
    print(f"Computed Understat features for {len(df)} matches")
    for col in feature_cols:
        non_null = df[col].notna().sum()
        mean_val = df[col].mean() if df[col].notna().any() else 0
        print(f"  {col:30s} non-null={non_null}/{len(df)}  mean={mean_val:.3f}")
    return df


def _get_formation_xg(season, team, formation, formation_data, team_most_used):
    """Look up formation xG stats for a team in a season.

    Priority:
    1. Current season + exact formation
    2. Current season + most-used formation (fallback)
    3. Prior season + exact formation
    4. Prior season + most-used formation
    5. None
    """
    # Try current season with exact formation
    if formation:
        key = (season, team, formation)
        if key in formation_data:
            return formation_data[key]

    # Try current season with most-used formation
    tk = (season, team)
    if tk in team_most_used:
        fallback_form = team_most_used[tk][0]
        key = (season, team, fallback_form)
        if key in formation_data:
            return formation_data[key]

    # Try prior season
    prior = _prior_season_label(season)
    if prior:
        if formation:
            key = (prior, team, formation)
            if key in formation_data:
                return formation_data[key]
        tk = (prior, team)
        if tk in team_most_used:
            fallback_form = team_most_used[tk][0]
            key = (prior, team, fallback_form)
            if key in formation_data:
                return formation_data[key]

    return None


def _get_top3_xg(season, team, player_xg_lookup):
    """Get avg xg_per90 and xa_per90 of top 3 players for a team.

    Tries current season first, falls back to prior season.
    Returns (avg_xg_per90, avg_xa_per90) or (None, None).
    """
    key = (season, team)
    players = player_xg_lookup.get(key, [])

    if len(players) < 3:
        # Fall back to prior season
        prior = _prior_season_label(season)
        if prior:
            players = player_xg_lookup.get((prior, team), [])

    if len(players) < 3:
        return (None, None)

    top3 = players[:3]
    avg_xg = float(np.mean([p[1] for p in top3]))
    avg_xa = float(np.mean([p[2] for p in top3]))
    return (round(avg_xg, 4), round(avg_xa, 4))


# ---------------------------------------------------------------------------
# Starting XI lineup features
# ---------------------------------------------------------------------------

def _build_player_name_index(conn, team_name, season):
    """Build a lookup of all Understat players for a team-season.

    Returns:
        all_players: list of dicts with player_name, xg_per90, xa_per90, minutes
        name_list: list of lowercase player names (for fuzzy matching)
    """
    rows = conn.execute("""
        SELECT player_name, xg_per90, xa_per90, minutes
        FROM understat_player_stats
        WHERE team_name = ? AND season = ?
        ORDER BY minutes DESC
    """, (team_name, season)).fetchall()

    all_players = []
    for name, xg, xa, mins in rows:
        all_players.append({
            "player_name": name,
            "xg_per90": xg,
            "xa_per90": xa,
            "minutes": mins,
        })
    return all_players


def _normalize_unicode(s):
    """Strip accents and normalize unicode for fuzzy comparison.

    Handles: é->e, ü->u, ö->o, ß->ss, plus U+FFFD replacement chars.
    """
    # Replace common special chars first
    s = s.replace("ß", "ss").replace("\ufffd", "")
    # NFD decompose then strip combining marks
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def fuzzy_match_player(input_name, team_players):
    """Match a user-provided player name against Understat player records.

    Uses multiple strategies with unicode normalization:
    1. Exact last-name match (highest precision for single-word inputs)
    2. Exact substring match (case-insensitive, unicode-normalized)
    3. difflib fuzzy match with cutoff=0.6

    Args:
        input_name: user-provided player name (e.g. "Kane", "Musiala", "Upamecano")
        team_players: list of dicts from _build_player_name_index

    Returns:
        matched player dict or None
    """
    if not team_players or not input_name:
        return None

    input_norm = _normalize_unicode(input_name.strip().lower())
    db_names_norm = [_normalize_unicode(p["player_name"].lower()) for p in team_players]

    # Strategy 1: exact full-name match (normalized)
    for i, db_norm in enumerate(db_names_norm):
        if input_norm == db_norm:
            return team_players[i]

    # Strategy 2: exact last-name match (single word input -> match last word of DB name)
    # This prevents "Kim" from matching "Kimmich" via substring
    if " " not in input_norm:
        for i, db_norm in enumerate(db_names_norm):
            db_parts = db_norm.split()
            # Match last name, or hyphenated last part (e.g. "Min-Jae")
            if db_parts and db_parts[-1] == input_norm:
                return team_players[i]
            # Also check first name (e.g., "Emre" in "Emre Can")
            if db_parts and db_parts[0] == input_norm:
                return team_players[i]

    # Strategy 3: multi-word exact substring (handles "Emre Can" in "Emre Can")
    if " " in input_norm:
        for i, db_norm in enumerate(db_names_norm):
            if input_norm == db_norm or input_norm in db_norm or db_norm in input_norm:
                return team_players[i]

    # Strategy 4: single-word substring (handles "Kane" in "Harry Kane")
    if " " not in input_norm and len(input_norm) >= 4:
        candidates = []
        for i, db_norm in enumerate(db_names_norm):
            if input_norm in db_norm:
                candidates.append(i)
        if len(candidates) == 1:
            return team_players[candidates[0]]
        # If multiple matches, prefer the one where input matches a whole word
        for i in candidates:
            if input_norm in db_names_norm[i].split():
                return team_players[i]
        if candidates:
            return team_players[candidates[0]]

    # Strategy 5: difflib fuzzy matching on normalized names
    matches = get_close_matches(input_norm, db_names_norm, n=1, cutoff=0.6)
    if matches:
        idx = db_names_norm.index(matches[0])
        return team_players[idx]

    return None


def get_xi_xg_features(xi_players, team_name, season, conn):
    """Compute top-3 xG/xA features from a declared starting XI.

    Args:
        xi_players: list of 11 player name strings
        team_name: canonical team name
        season: season label (e.g. "2025/26")
        conn: sqlite3 connection

    Returns:
        (top3_xg_per90, top3_xa_per90, match_details)
        match_details: list of dicts with input_name, matched_name, xg_per90, xa_per90, status
    """
    from features.player_features import _prior_season_label

    team_players = _build_player_name_index(conn, team_name, season)

    # Also load prior season for fallback
    prior = _prior_season_label(season)
    prior_players = _build_player_name_index(conn, team_name, prior) if prior else []

    matched = []
    details = []
    for name in xi_players:
        if not name or not name.strip():
            continue

        # Try current season first
        player = fuzzy_match_player(name, team_players)
        source = "current_season"

        # Fall back to prior season
        if player is None and prior_players:
            player = fuzzy_match_player(name, prior_players)
            source = "prior_season"

        if player and player["minutes"] >= 90:
            matched.append(player)
            details.append({
                "input_name": name,
                "matched_name": player["player_name"],
                "xg_per90": player["xg_per90"],
                "xa_per90": player["xa_per90"],
                "minutes": player["minutes"],
                "status": f"matched ({source})",
            })
        else:
            details.append({
                "input_name": name,
                "matched_name": None,
                "xg_per90": None,
                "xa_per90": None,
                "minutes": None,
                "status": "not_found" if player is None else "below_90min",
            })

    # If fewer than 6 matched, fall back to team season average
    if len(matched) < 6:
        # Use _get_top3_xg team-level fallback
        player_xg_lookup = {}
        # Build quick lookup from team_players
        qualified = [p for p in team_players if p["minutes"] >= 450]
        if len(qualified) >= 3:
            key = (season, team_name)
            player_xg_lookup[key] = [
                (p["player_name"], p["xg_per90"], p["xa_per90"]) for p in qualified
            ]
        elif prior_players:
            qualified_prior = [p for p in prior_players if p["minutes"] >= 450]
            if len(qualified_prior) >= 3:
                key = (prior, team_name)
                player_xg_lookup[key] = [
                    (p["player_name"], p["xg_per90"], p["xa_per90"]) for p in qualified_prior
                ]

        top3 = _get_top3_xg(season, team_name, player_xg_lookup)
        for d in details:
            if d["status"] in ("not_found", "below_90min"):
                d["status"] += " (team avg fallback)"
        return top3[0], top3[1], details

    # Compute from matched XI players: top 3 by xg_per90
    sorted_by_xg = sorted(matched, key=lambda p: p["xg_per90"], reverse=True)
    top3 = sorted_by_xg[:3]
    avg_xg = float(np.mean([p["xg_per90"] for p in top3]))
    avg_xa = float(np.mean([p["xa_per90"] for p in top3]))

    return round(avg_xg, 4), round(avg_xa, 4), details


def get_typical_xi(team_name, season, db_path=DB_PATH):
    """Get the most frequently used starting XI for a team in a season.

    Parses all lineups for the team and counts player appearances in startXI.
    Returns the 11 most frequent starters by full name from api-football.

    Args:
        team_name: canonical team name
        season: season label (e.g. "2025/26")

    Returns:
        list of 11 player name strings (most frequent starters),
        or empty list if no data
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT l.raw_json, m.home_team, m.away_team
        FROM lineups l
        JOIN matches m ON l.match_id = m.match_id
        WHERE (m.home_team = ? OR m.away_team = ?) AND m.season = ?
    """, (team_name, team_name, season)).fetchall()
    conn.close()

    if not rows:
        return []

    from collections import Counter
    player_counter = Counter()

    for raw_json, home_team, away_team in rows:
        d = json.loads(raw_json)
        side_idx = 0 if home_team == team_name else 1
        try:
            for entry in d[side_idx].get("startXI", []):
                name = entry.get("player", {}).get("name", "")
                if name:
                    player_counter[name] += 1
        except (IndexError, KeyError, TypeError):
            continue

    # Return top 11 by frequency
    return [name for name, _ in player_counter.most_common(11)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "understat":
        df = compute_understat_features()
    else:
        df = compute_squad_features()
    if not df.empty:
        print(f"\nSample rows:")
        print(df.head(10).to_string(index=False))
