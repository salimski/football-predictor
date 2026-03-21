"""
Feature engineering for the Bundesliga over/under 2.5 goals classifier.

All rolling features use strictly prior matches (match_date < current)
to avoid any data leakage.
"""

import sqlite3
import json
import re
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SEASONS
from collector.normalize import normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAT_NAMES = [
    "shots_on_target",   # [0]
    "shots_off_target",  # [1]
    "shots_total",       # [2]
    "shots_blocked",     # [3]
    "shots_inside_box",  # [4]
    "shots_outside_box", # [5]
    "fouls",             # [6]
    "corners",           # [7]
    "offsides",          # [8]
    "possession",        # [9]
    "yellow_cards",      # [10]
    "red_cards",         # [11]
    "saves",             # [12]
    "passes_total",      # [13]
    "passes_accurate",   # [14]
    "pass_pct",          # [15]
]

SEASON_ORDER = [s["label"] for s in SEASONS]

ROLL_COLS = [
    "games_available", "goals_scored", "goals_conceded", "total_goals",
    "over25_rate", "xg_for", "xg_against", "shots_on_target", "shots_total",
    "corners", "possession", "wins", "draws",
]

VENUE_ROLL_COLS = [
    "venue_games", "venue_goals_scored", "venue_goals_conceded",
    "venue_xg_for", "venue_xg_against",
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_raw_tables(db_path):
    conn = sqlite3.connect(db_path)
    df_matches   = pd.read_sql("SELECT * FROM matches",    conn)
    df_stats     = pd.read_sql("SELECT * FROM statistics", conn)
    df_lineups   = pd.read_sql("SELECT * FROM lineups",    conn)
    df_standings = pd.read_sql("SELECT * FROM standings",  conn)
    df_xg        = pd.read_sql("SELECT * FROM xg",         conn)
    # odds table (may not exist in older DBs)
    try:
        df_odds  = pd.read_sql("SELECT * FROM odds",       conn)
    except Exception:
        df_odds  = pd.DataFrame()
    conn.close()
    return df_matches, df_stats, df_lineups, df_standings, df_xg, df_odds


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def parse_round(s):
    m = re.search(r"(\d+)$", str(s or ""))
    return int(m.group(1)) if m else None


def parse_matches(df):
    records = []
    for _, row in df.iterrows():
        d = json.loads(row["raw_json"])
        fixture  = d.get("fixture", {})
        league   = d.get("league", {})
        goals    = d.get("goals", {})
        halftime = d.get("score", {}).get("halftime", {})
        date_str = fixture.get("date", "")
        records.append({
            "match_id":         row["match_id"],
            "season":           row["season"],
            "match_date":       date_str[:10] if date_str else None,
            "round_number":     parse_round(league.get("round", "")),
            "home_team":        row["home_team"],
            "away_team":        row["away_team"],
            "goals_home":       goals.get("home"),
            "goals_away":       goals.get("away"),
            "pm_halftime_home": halftime.get("home"),
            "pm_halftime_away": halftime.get("away"),
        })
    return pd.DataFrame(records)


def _safe_numeric(val):
    if val is None:
        return None
    s = str(val).replace("%", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_stats(df):
    records = []
    for _, row in df.iterrows():
        d = json.loads(row["raw_json"])
        rec = {"match_id": row["match_id"]}
        for side_idx, prefix in [(0, "pm_h_"), (1, "pm_a_")]:
            try:
                stats = d[side_idx]["statistics"]
            except (IndexError, KeyError, TypeError):
                stats = []
            for i, name in enumerate(STAT_NAMES):
                val = _safe_numeric(stats[i]["value"]) if i < len(stats) else None
                rec[f"{prefix}{name}"] = val
        records.append(rec)
    return pd.DataFrame(records)


def _parse_formation_parts(formation):
    """Parse '4-2-3-1' -> (defenders=4, midfielders=5, forwards=1)."""
    if not formation or not isinstance(formation, str):
        return None, None, None
    parts = formation.split("-")
    if len(parts) < 2:
        return None, None, None
    try:
        defenders   = int(parts[0])
        midfielders = sum(int(p) for p in parts[1:-1]) if len(parts) > 2 else int(parts[1])
        forwards    = int(parts[-1])
        return defenders, midfielders, forwards
    except (ValueError, TypeError):
        return None, None, None


def parse_lineups(df):
    records = []
    for _, row in df.iterrows():
        d = json.loads(row["raw_json"])
        h_form = a_form = None
        try:
            h_form = d[0].get("formation")
            a_form = d[1].get("formation")
        except (IndexError, KeyError, TypeError):
            pass

        h_def, h_mid, h_fwd = _parse_formation_parts(h_form)
        a_def, a_mid, a_fwd = _parse_formation_parts(a_form)
        records.append({
            "match_id":                row["match_id"],
            "h_formation":             h_form,
            "a_formation":             a_form,
            "h_formation_defenders":   h_def,
            "h_formation_midfielders": h_mid,
            "h_formation_forwards":    h_fwd,
            "a_formation_defenders":   a_def,
            "a_formation_midfielders": a_mid,
            "a_formation_forwards":    a_fwd,
        })
    return pd.DataFrame(records)


def parse_xg(df):
    records = []
    for _, row in df.iterrows():
        rec = {
            "match_date":           row["match_date"],
            "home_team":            row["home_team"],
            "away_team":            row["away_team"],
            "pm_home_xg":           row["home_xg"] if pd.notna(row["home_xg"]) else None,
            "pm_away_xg":           row["away_xg"] if pd.notna(row["away_xg"]) else None,
            "xg_forecast_home_win": None,
            "xg_forecast_draw":     None,
            "xg_forecast_away_win": None,
        }
        raw = row["raw_json"]
        if pd.notna(raw):
            try:
                d        = json.loads(raw)
                forecast = d.get("forecast") or {}
                if forecast:
                    rec["xg_forecast_home_win"] = _safe_numeric(forecast.get("w"))
                    rec["xg_forecast_draw"]     = _safe_numeric(forecast.get("d"))
                    rec["xg_forecast_away_win"] = _safe_numeric(forecast.get("l"))
            except (json.JSONDecodeError, TypeError):
                pass
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def build_team_view(df):
    """2 rows per match (home+away perspective) with venue tag."""
    rows = []
    for _, r in df.iterrows():
        gh = r["goals_home"]
        ga = r["goals_away"]
        tg = (gh + ga) if (pd.notna(gh) and pd.notna(ga)) else None

        rows.append({
            "match_id": r["match_id"], "match_date": r["match_date"],
            "team": r["home_team"], "venue": "home",
            "goals_scored": gh, "goals_conceded": ga, "total_goals": tg,
            "xg_for": r.get("pm_home_xg"), "xg_against": r.get("pm_away_xg"),
            "shots_on_target": r.get("pm_h_shots_on_target"),
            "shots_total": r.get("pm_h_shots_total"),
            "corners": r.get("pm_h_corners"),
            "possession": r.get("pm_h_possession"),
        })
        rows.append({
            "match_id": r["match_id"], "match_date": r["match_date"],
            "team": r["away_team"], "venue": "away",
            "goals_scored": ga, "goals_conceded": gh, "total_goals": tg,
            "xg_for": r.get("pm_away_xg"), "xg_against": r.get("pm_home_xg"),
            "shots_on_target": r.get("pm_a_shots_on_target"),
            "shots_total": r.get("pm_a_shots_total"),
            "corners": r.get("pm_a_corners"),
            "possession": r.get("pm_a_possession"),
        })

    tv = pd.DataFrame(rows)
    tv["match_date"] = pd.to_datetime(tv["match_date"])
    return tv


def compute_rolling_for_team(team, before_date, team_view):
    prior = team_view[
        (team_view["team"] == team) &
        (team_view["match_date"] < before_date)
    ].sort_values("match_date").tail(5)

    n = len(prior)
    if n == 0:
        return {col: (0 if col == "games_available" else None) for col in ROLL_COLS}

    def _mean(col):
        s = prior[col]
        return float(s.mean(skipna=True)) if s.notna().any() else None

    mask = prior["goals_scored"].notna() & prior["goals_conceded"].notna()
    if mask.any():
        scored   = prior.loc[mask, "goals_scored"]
        conceded = prior.loc[mask, "goals_conceded"]
        wins  = float((scored > conceded).mean())
        draws = float((scored == conceded).mean())
    else:
        wins = draws = 0.0

    tg = prior["total_goals"]
    over25_rate = float((tg.dropna() > 2.5).mean()) if tg.notna().any() else None

    return {
        "games_available": n,
        "goals_scored":    _mean("goals_scored"),
        "goals_conceded":  _mean("goals_conceded"),
        "total_goals":     _mean("total_goals"),
        "over25_rate":     over25_rate,
        "xg_for":          _mean("xg_for"),
        "xg_against":      _mean("xg_against"),
        "shots_on_target": _mean("shots_on_target"),
        "shots_total":     _mean("shots_total"),
        "corners":         _mean("corners"),
        "possession":      _mean("possession"),
        "wins":            wins,
        "draws":           draws,
    }


def compute_venue_rolling(team, venue, before_date, team_view):
    """Rolling stats for a team at a specific venue (home or away), last 5."""
    prior = team_view[
        (team_view["team"] == team) &
        (team_view["venue"] == venue) &
        (team_view["match_date"] < before_date)
    ].sort_values("match_date").tail(5)

    n = len(prior)
    if n == 0:
        return {
            "venue_games": 0, "venue_goals_scored": None, "venue_goals_conceded": None,
            "venue_xg_for": None, "venue_xg_against": None,
        }

    def _vmean(col):
        s = prior[col]
        return float(s.mean(skipna=True)) if s.notna().any() else None

    return {
        "venue_games":          n,
        "venue_goals_scored":   _vmean("goals_scored"),
        "venue_goals_conceded": _vmean("goals_conceded"),
        "venue_xg_for":         _vmean("xg_for"),
        "venue_xg_against":     _vmean("xg_against"),
    }


def add_rolling_features(df, team_view):
    """Compute all rolling features: overall + venue-specific."""
    h_roll = []
    a_roll = []
    h_venue = []
    a_venue = []

    for _, row in df.iterrows():
        before_date = pd.Timestamp(row["match_date"])
        h_roll.append(compute_rolling_for_team(row["home_team"], before_date, team_view))
        a_roll.append(compute_rolling_for_team(row["away_team"], before_date, team_view))
        h_venue.append(compute_venue_rolling(row["home_team"], "home", before_date, team_view))
        a_venue.append(compute_venue_rolling(row["away_team"], "away", before_date, team_view))

    for col in ROLL_COLS:
        df[f"h_roll_{col}"] = [r[col] for r in h_roll]
        df[f"a_roll_{col}"] = [r[col] for r in a_roll]

    for col in VENUE_ROLL_COLS:
        df[f"h_{col}"] = [r[col] for r in h_venue]
        df[f"a_{col}"] = [r[col] for r in a_venue]

    return df


# ---------------------------------------------------------------------------
# Formation matchup & Head-to-head
# ---------------------------------------------------------------------------

def add_formation_matchup(df):
    """For each match, compute avg total goals in prior matches with same formation pair."""
    df = df.copy()
    df["match_date_ts"] = pd.to_datetime(df["match_date"])
    df_sorted = df.sort_values("match_date_ts").reset_index(drop=True)

    fm_avg = []
    # Build a running history by formation pair
    history = {}  # (h_form, a_form) -> list of total_goals

    for _, row in df_sorted.iterrows():
        key = (row["h_formation"], row["a_formation"])
        if key[0] is not None and key[1] is not None and key in history and len(history[key]) > 0:
            fm_avg.append(float(np.mean(history[key])))
        else:
            fm_avg.append(None)
        # Add this match to history
        if key[0] is not None and key[1] is not None and pd.notna(row.get("total_goals")):
            history.setdefault(key, []).append(row["total_goals"])

    df_sorted["formation_matchup_avg_goals"] = fm_avg

    # Re-merge back to original order
    result = df[["match_id"]].merge(
        df_sorted[["match_id", "formation_matchup_avg_goals"]], on="match_id", how="left"
    )
    df["formation_matchup_avg_goals"] = result["formation_matchup_avg_goals"].values
    df = df.drop(columns=["match_date_ts"], errors="ignore")
    return df


def add_h2h(df):
    """For each match, compute avg total goals in prior meetings of the same two teams."""
    df = df.copy()
    df["match_date_ts"] = pd.to_datetime(df["match_date"])
    df_sorted = df.sort_values("match_date_ts").reset_index(drop=True)

    h2h_avg = []
    history = {}  # frozenset({team1, team2}) -> list of total_goals

    for _, row in df_sorted.iterrows():
        key = frozenset([row["home_team"], row["away_team"]])
        if key in history and len(history[key]) > 0:
            h2h_avg.append(float(np.mean(history[key])))
        else:
            h2h_avg.append(None)
        if pd.notna(row.get("total_goals")):
            history.setdefault(key, []).append(row["total_goals"])

    df_sorted["h2h_avg_goals"] = h2h_avg

    result = df[["match_id"]].merge(
        df_sorted[["match_id", "h2h_avg_goals"]], on="match_id", how="left"
    )
    df["h2h_avg_goals"] = result["h2h_avg_goals"].values
    df = df.drop(columns=["match_date_ts"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Tier features
# ---------------------------------------------------------------------------

def _build_standings_rank_lookup(df_standings):
    """Return {season: {canonical_team: rank}}."""
    lookup = {}
    for _, row in df_standings.iterrows():
        d = json.loads(row["raw_json"])
        try:
            table = d[0]["league"]["standings"][0]
        except (KeyError, IndexError, TypeError):
            continue
        lookup[row["season"]] = {
            normalize(entry["team"]["name"]): entry["rank"]
            for entry in table
        }
    return lookup


def add_tier_features(df, df_standings):
    """Compute tier scores based on sum of ranks over last 3 completed seasons."""
    rank_lookup = _build_standings_rank_lookup(df_standings)

    def _get_prior_seasons(match_season, n=3):
        try:
            idx = SEASON_ORDER.index(match_season)
        except ValueError:
            return []
        start = max(0, idx - n)
        return SEASON_ORDER[start:idx]

    def _tier_score(team, prior_seasons):
        total = 0
        for i in range(3):
            if i < len(prior_seasons):
                total += rank_lookup.get(prior_seasons[i], {}).get(team, 19)
            else:
                total += 19
        return total

    h_scores = []
    a_scores = []
    for _, row in df.iterrows():
        prior = _get_prior_seasons(row["season"], n=3)
        h_scores.append(_tier_score(row["home_team"], prior))
        a_scores.append(_tier_score(row["away_team"], prior))

    df["h_tier_score"] = h_scores
    df["a_tier_score"] = a_scores
    return df


# ---------------------------------------------------------------------------
# Formation extras
# ---------------------------------------------------------------------------

def add_xg_forecast(df):
    """Compute pre-match xG forecasts from venue-specific rolling xG averages.

    h_xg_forecast = (h_venue_xg_for + a_venue_xg_against) / 2
    a_xg_forecast = (a_venue_xg_for + h_venue_xg_against) / 2
    total_xg_forecast = h + a
    """
    h_venue_xg_for     = df["h_venue_xg_for"]
    a_venue_xg_against = df["a_venue_xg_against"]
    a_venue_xg_for     = df["a_venue_xg_for"]
    h_venue_xg_against = df["h_venue_xg_against"]

    df["h_xg_forecast"]     = (h_venue_xg_for + a_venue_xg_against) / 2
    df["a_xg_forecast"]     = (a_venue_xg_for + h_venue_xg_against) / 2
    df["total_xg_forecast"] = df["h_xg_forecast"] + df["a_xg_forecast"]
    return df


def add_formation_extras(df):
    """Forward surplus and defensive surplus."""
    df["fwd_vs_def"] = df["h_formation_forwards"] - df["a_formation_defenders"]
    df["def_vs_fwd"] = df["h_formation_defenders"] - df["a_formation_forwards"]
    return df


# ---------------------------------------------------------------------------
# Standings features (prior season detail)
# ---------------------------------------------------------------------------

def build_standings_lookup(df_standings):
    lookup = {}
    for _, row in df_standings.iterrows():
        season = row["season"]
        d = json.loads(row["raw_json"])
        try:
            table = d[0]["league"]["standings"][0]
        except (KeyError, IndexError, TypeError):
            continue
        team_dict = {}
        for entry in table:
            raw_name  = entry.get("team", {}).get("name", "")
            canonical = normalize(raw_name)
            all_s     = entry.get("all", {})
            team_dict[canonical] = {
                "rank":          entry.get("rank"),
                "points":        entry.get("points"),
                "goals_for":     all_s.get("goals", {}).get("for"),
                "goals_against": all_s.get("goals", {}).get("against"),
                "goal_diff":     entry.get("goalsDiff"),
            }
        lookup[season] = team_dict
    return lookup


def _prior_season(current_season):
    try:
        idx = SEASON_ORDER.index(current_season)
    except ValueError:
        return None
    return SEASON_ORDER[idx - 1] if idx > 0 else None


def add_standings_features(df, lookup):
    h_rank = []; h_pts = []; h_gf = []; h_ga = []; h_gd = []
    a_rank = []; a_pts = []; a_gf = []; a_ga = []; a_gd = []
    rank_diff = []

    for _, row in df.iterrows():
        prev       = _prior_season(row["season"])
        prev_table = lookup.get(prev, {}) if prev else {}
        h_data     = prev_table.get(row["home_team"], {})
        a_data     = prev_table.get(row["away_team"], {})

        h_rank.append(h_data.get("rank"))
        h_pts.append(h_data.get("points"))
        h_gf.append(h_data.get("goals_for"))
        h_ga.append(h_data.get("goals_against"))
        h_gd.append(h_data.get("goal_diff"))
        a_rank.append(a_data.get("rank"))
        a_pts.append(a_data.get("points"))
        a_gf.append(a_data.get("goals_for"))
        a_ga.append(a_data.get("goals_against"))
        a_gd.append(a_data.get("goal_diff"))

        hr = h_data.get("rank")
        ar = a_data.get("rank")
        rank_diff.append((hr - ar) if (hr is not None and ar is not None) else None)

    df["h_standing_rank"]          = h_rank
    df["h_standing_points"]        = h_pts
    df["h_standing_goals_for"]     = h_gf
    df["h_standing_goals_against"] = h_ga
    df["h_standing_goal_diff"]     = h_gd
    df["a_standing_rank"]          = a_rank
    df["a_standing_points"]        = a_pts
    df["a_standing_goals_for"]     = a_gf
    df["a_standing_goals_against"] = a_ga
    df["a_standing_goal_diff"]     = a_gd
    df["standing_rank_diff"]       = rank_diff
    return df


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def add_targets(df):
    df["total_goals"]   = df["goals_home"] + df["goals_away"]
    df["target_over25"] = (df["total_goals"] > 2).astype(int)
    df["target_under35"] = (df["total_goals"] <= 3).astype(int)
    df["target_under45"] = (df["total_goals"] <= 4).astype(int)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def parse_odds(df_odds):
    """Extract odds columns for join onto features."""
    cols = ["match_date", "home_team", "away_team",
            "b365_prob_over25", "bb_avg_prob_over25",
            "pinnacle_prob_over25"]
    if len(df_odds) == 0:
        return pd.DataFrame(columns=cols)
    available = [c for c in cols if c in df_odds.columns]
    out = df_odds[available].copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out


def build(db_path):
    print("Loading raw tables...")
    df_matches, df_stats, df_lineups, df_standings, df_xg, df_odds = load_raw_tables(db_path)

    print("Parsing matches...")
    df = parse_matches(df_matches)

    print("Parsing statistics...")
    df = df.merge(parse_stats(df_stats), on="match_id", how="left")

    print("Parsing lineups...")
    df = df.merge(parse_lineups(df_lineups), on="match_id", how="left")

    print("Parsing xg...")
    df_xg_parsed = parse_xg(df_xg)
    df = df.merge(df_xg_parsed, on=["match_date", "home_team", "away_team"], how="left")

    print("Parsing odds...")
    df_odds_parsed = parse_odds(df_odds)
    df = df.merge(df_odds_parsed, on=["match_date", "home_team", "away_team"], how="left")

    print("Adding targets...")
    df = add_targets(df)

    print("Building team view...")
    team_view = build_team_view(df)

    print("Computing rolling features (overall + venue-specific)...")
    df = add_rolling_features(df, team_view)

    print("Adding formation matchup avg goals...")
    df = add_formation_matchup(df)

    print("Adding head-to-head avg goals...")
    df = add_h2h(df)

    print("Adding tier features...")
    df = add_tier_features(df, df_standings)

    print("Adding pre-match xG forecasts...")
    df = add_xg_forecast(df)

    print("Adding formation extras...")
    df = add_formation_extras(df)

    print("Building standings lookup...")
    standings_lookup = build_standings_lookup(df_standings)

    print("Adding standings features...")
    df = add_standings_features(df, standings_lookup)

    # Squad rating features (from player_stats table)
    try:
        from features.player_features import compute_squad_features
        print("Computing squad rating features...")
        squad_df = compute_squad_features(db_path)
        if not squad_df.empty:
            df = df.merge(squad_df, on="match_id", how="left")
            print(f"  Merged squad features: {squad_df.columns.tolist()}")
        else:
            print("  No squad features available (player_stats table empty)")
            for col in ["h_squad_avg_rating", "a_squad_avg_rating",
                        "h_squad_depth_rating", "a_squad_depth_rating"]:
                df[col] = np.nan
    except Exception as exc:
        print(f"  [WARN] Squad features skipped: {exc}")
        for col in ["h_squad_avg_rating", "a_squad_avg_rating",
                    "h_squad_depth_rating", "a_squad_depth_rating"]:
            df[col] = np.nan

    return df


def write_features(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Written {len(df)} rows, {len(df.columns)} columns to features table.")
