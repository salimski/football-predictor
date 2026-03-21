"""
Predict over/under 2.5 goals for an upcoming match.

Two-layer architecture:
  Layer 1: Dixon-Coles  -> lambda_home, lambda_away, scoreline probs
  Layer 2: XGBoost      -> calibrated P(over 2.5) using DC + context features

Usage:
    python predict.py
    (Interactive - prompts for match details)
"""

import sqlite3
import json
import os
import sys

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from config import DB_PATH, SEASONS
from collector.normalize import normalize
from features.builder import (
    build_team_view, compute_rolling_for_team, compute_venue_rolling,
    _parse_formation_parts, _build_standings_rank_lookup,
    load_raw_tables, parse_matches, parse_stats, parse_lineups, parse_xg,
    add_targets, SEASON_ORDER,
)
from model.dixon_coles import DixonColesModel, scoreline_matrix, marginalize_goals
from features.player_features import (
    _load_formation_lookup, _load_player_xg_lookup,
    _get_formation_xg, _get_top3_xg,
    get_xi_xg_features, get_typical_xi,
)

MODEL_DIR = "model/saved"


def load_models():
    """Load Dixon-Coles and all XGBoost models (O/U 2.5, U3.5, U4.5)."""
    dc = DixonColesModel.load(os.path.join(MODEL_DIR, "dixon_coles.json"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    xgb_u35 = joblib.load(os.path.join(MODEL_DIR, "xgb_under35.joblib"))
    xgb_u45 = joblib.load(os.path.join(MODEL_DIR, "xgb_under45.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    # Ensure per-model feature lists exist (backward compat)
    if "features_u35" not in meta:
        meta["features_u35"] = meta["features"]
    if "features_u45" not in meta:
        meta["features_u45"] = meta["features"]
    return dc, xgb_model, xgb_u35, xgb_u45, meta


def build_current_data(db_path):
    """Load and build the full feature set from the database (for lookups)."""
    df_matches, df_stats, df_lineups, df_standings, df_xg, _df_odds = load_raw_tables(db_path)
    df = parse_matches(df_matches)
    df = df.merge(parse_stats(df_stats), on="match_id", how="left")
    df = df.merge(parse_lineups(df_lineups), on="match_id", how="left")
    df_xg_parsed = parse_xg(df_xg)
    df = df.merge(df_xg_parsed, on=["match_date", "home_team", "away_team"], how="left")
    df = add_targets(df)
    team_view = build_team_view(df)
    return df, team_view, df_standings


def load_understat_lookups(db_path):
    """Load Understat formation and player xG lookups for prediction time."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    formation_data, team_most_used = _load_formation_lookup(conn)
    player_xg_lookup = _load_player_xg_lookup(conn)
    conn.close()
    return formation_data, team_most_used, player_xg_lookup


def build_feature_row(home_team, away_team, h_formation, a_formation,
                      df, team_view, df_standings, dc_model, features_list,
                      b365_over_odds=None, pinnacle_over_odds=None,
                      b365_under35_odds=None, b365_under45_odds=None,
                      home_xi=None, away_xi=None):
    """Build a single-row DataFrame with all model features (context + DC).

    Args:
        b365_over_odds: Optional Bet365 decimal odds for over 2.5 goals
            (e.g. 1.85). If provided, computes implied probability and
            odds_dc_diff. If None, odds features are NaN.
        pinnacle_over_odds: Optional Pinnacle decimal odds for over 2.5
            goals. If provided, computes pinnacle_prob_over25 and
            pinnacle_dc_diff. If None, both are NaN.
        b365_under35_odds: Optional Bet365 decimal odds for under 3.5 goals
            (e.g. 1.45). Converted: P(over 3.5) = 1 - 1/odds.
        b365_under45_odds: Optional Bet365 decimal odds for under 4.5 goals
            (e.g. 1.20). Converted: P(over 4.5) = 1 - 1/odds.
        home_xi: Optional list of 11 player name strings for home team.
            When provided, top-3 xG/xA features are computed from the
            declared lineup instead of team season averages.
        away_xi: Optional list of 11 player name strings for away team.
    """
    now = pd.Timestamp.now()

    # Venue-specific rolling (for xG forecasts)
    h_venue = compute_venue_rolling(home_team, "home", now, team_view)
    a_venue = compute_venue_rolling(away_team, "away", now, team_view)

    # Pre-match xG forecasts from venue-specific rolling
    h_venue_xg_for = h_venue.get("venue_xg_for")
    a_venue_xg_against = a_venue.get("venue_xg_against")
    a_venue_xg_for = a_venue.get("venue_xg_for")
    h_venue_xg_against = h_venue.get("venue_xg_against")

    h_xg_forecast = None
    a_xg_forecast = None
    total_xg_forecast = None
    if h_venue_xg_for is not None and a_venue_xg_against is not None:
        h_xg_forecast = (h_venue_xg_for + a_venue_xg_against) / 2
    if a_venue_xg_for is not None and h_venue_xg_against is not None:
        a_xg_forecast = (a_venue_xg_for + h_venue_xg_against) / 2
    if h_xg_forecast is not None and a_xg_forecast is not None:
        total_xg_forecast = h_xg_forecast + a_xg_forecast

    # Formation parsing
    h_def, h_mid, h_fwd = _parse_formation_parts(h_formation)
    a_def, a_mid, a_fwd = _parse_formation_parts(a_formation)

    # Dixon-Coles predictions (Layer 1)
    dc_pred = dc_model.predict_match(home_team, away_team)

    # Odds features
    b365_prob_over25 = None
    bb_avg_prob_over25 = None
    odds_dc_diff = None
    if b365_over_odds is not None and b365_over_odds > 0:
        b365_prob_over25 = 1.0 / b365_over_odds
        bb_avg_prob_over25 = b365_prob_over25  # B365 as proxy for market avg
        odds_dc_diff = b365_prob_over25 - dc_pred["prob_over25"]

    # Pinnacle features
    pinnacle_prob_over25 = None
    pinnacle_dc_diff = None
    if pinnacle_over_odds is not None and pinnacle_over_odds > 0:
        pinnacle_prob_over25 = 1.0 / pinnacle_over_odds
        pinnacle_dc_diff = pinnacle_prob_over25 - dc_pred["prob_over25"]

    # 3.5 line odds: under odds → derive over probability
    b365_prob_over35 = None
    odds_dc_diff_35 = None
    odds_source_35 = "DC only"
    if b365_under35_odds is not None and b365_under35_odds > 1.0:
        b365_prob_under35 = 1.0 / b365_under35_odds
        b365_prob_over35 = 1.0 - b365_prob_under35
        odds_dc_diff_35 = b365_prob_over35 - dc_pred["prob_over35"]
        odds_source_35 = "live odds"

    # 4.5 line odds: under odds → derive over probability
    b365_prob_over45 = None
    odds_dc_diff_45 = None
    odds_source_45 = "DC only"
    if b365_under45_odds is not None and b365_under45_odds > 1.0:
        b365_prob_under45 = 1.0 / b365_under45_odds
        b365_prob_over45 = 1.0 - b365_prob_under45
        odds_dc_diff_45 = b365_prob_over45 - dc_pred["prob_over45"]
        odds_source_45 = "live odds"

    # 2.5 line odds source
    odds_source_25 = "live odds" if b365_prob_over25 is not None else "DC only"

    row = {
        # Formation (8 features)
        "h_formation_defenders": h_def,
        "h_formation_midfielders": h_mid,
        "h_formation_forwards": h_fwd,
        "a_formation_defenders": a_def,
        "a_formation_midfielders": a_mid,
        "a_formation_forwards": a_fwd,
        "fwd_vs_def": (h_fwd - a_def) if (h_fwd is not None and a_def is not None) else None,
        "def_vs_fwd": (h_def - a_fwd) if (h_def is not None and a_fwd is not None) else None,
        # Pre-match xG forecasts (3 features)
        "h_xg_forecast": h_xg_forecast,
        "a_xg_forecast": a_xg_forecast,
        "total_xg_forecast": total_xg_forecast,
        # Odds — 2.5 line (3 features, used by O/U 2.5 model)
        "b365_prob_over25": b365_prob_over25,
        "bb_avg_prob_over25": bb_avg_prob_over25,
        "odds_dc_diff": odds_dc_diff,
        # Odds — 3.5 line (2 features, used by U3.5 model)
        "b365_prob_over35": b365_prob_over35,
        "odds_dc_diff_35": odds_dc_diff_35,
        # Odds — 4.5 line (2 features, used by U4.5 model)
        "b365_prob_over45": b365_prob_over45,
        "odds_dc_diff_45": odds_dc_diff_45,
        # Pinnacle (2 features)
        "pinnacle_prob_over25": pinnacle_prob_over25,
        "pinnacle_dc_diff": pinnacle_dc_diff,
        # Dixon-Coles outputs (8 features)
        "dc_lambda_home": dc_pred["lambda_home"],
        "dc_lambda_away": dc_pred["lambda_away"],
        "dc_prob_over25": dc_pred["prob_over25"],
        "dc_prob_under25": dc_pred["prob_under25"],
        "dc_prob_over35": dc_pred["prob_over35"],
        "dc_prob_under35": dc_pred["prob_under35"],
        "dc_prob_over45": dc_pred["prob_over45"],
        "dc_prob_under45": dc_pred["prob_under45"],
    }

    # Understat formation xG features
    xi_details = {"home": None, "away": None}
    try:
        formation_data, team_most_used, player_xg_lookup = load_understat_lookups(DB_PATH)
        current_season = SEASONS[-1]["label"]  # 2025/26

        h_fxg = _get_formation_xg(current_season, home_team, h_formation,
                                   formation_data, team_most_used)
        a_fxg = _get_formation_xg(current_season, away_team, a_formation,
                                   formation_data, team_most_used)

        if h_fxg:
            row["h_formation_xg_per_game"] = h_fxg["xg_for_per_game"]
            row["h_formation_xga_per_game"] = h_fxg["xg_against_per_game"]
        if a_fxg:
            row["a_formation_xg_per_game"] = a_fxg["xg_for_per_game"]
            row["a_formation_xga_per_game"] = a_fxg["xg_against_per_game"]
        if h_fxg and a_fxg:
            row["formation_xg_matchup"] = (h_fxg["xg_for_per_game"] +
                                            a_fxg["xg_for_per_game"])

        # Top 3 player xG/xA features — XI-specific when lineup declared
        conn = sqlite3.connect(DB_PATH)
        if home_xi:
            h_xg, h_xa, h_details = get_xi_xg_features(
                home_xi, home_team, current_season, conn)
            xi_details["home"] = h_details
        else:
            h_xg, h_xa = _get_top3_xg(current_season, home_team, player_xg_lookup)
            h_details = None

        if away_xi:
            a_xg, a_xa, a_details = get_xi_xg_features(
                away_xi, away_team, current_season, conn)
            xi_details["away"] = a_details
        else:
            a_xg, a_xa = _get_top3_xg(current_season, away_team, player_xg_lookup)
            a_details = None
        conn.close()

        if h_xg is not None:
            row["h_top3_xg_per90"] = h_xg
            row["h_top3_xa_per90"] = h_xa
        if a_xg is not None:
            row["a_top3_xg_per90"] = a_xg
            row["a_top3_xa_per90"] = a_xa
    except Exception:
        pass  # Features will be NaN, XGBoost handles it

    X_full = pd.DataFrame([row])
    # Ensure all possible feature columns exist (superset of all model feature lists)
    all_needed = set(features_list)
    if isinstance(features_list, dict):
        for v in features_list.values():
            all_needed |= set(v)
    for col in all_needed:
        if col not in X_full.columns:
            X_full[col] = np.nan
    for col in X_full.columns:
        X_full[col] = pd.to_numeric(X_full[col], errors="coerce")

    odds_sources = {
        "o25": odds_source_25,
        "u35": odds_source_35,
        "u45": odds_source_45,
    }
    return X_full, dc_pred, xi_details, odds_sources


def get_team_names(df):
    return sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))


def _prompt_xi(team_name, season):
    """Prompt user for starting XI, with option to load typical XI."""
    print(f"\n  Starting XI for {team_name}:")
    use_xi = input("  Enter starting XI? (y/n, or 't' for typical XI): ").strip().lower()

    if use_xi == "t":
        typical = get_typical_xi(team_name, season)
        if typical:
            print(f"  Loaded typical XI ({len(typical)} players):")
            for i, name in enumerate(typical, 1):
                print(f"    {i:>2}. {name}")
            return typical
        else:
            print("  No lineup data found for this team/season.")
            return None

    if use_xi != "y":
        return None

    xi = []
    print("  Enter 11 player names (one per line):")
    for i in range(1, 12):
        name = input(f"    Player {i:>2}: ").strip()
        xi.append(name)
    return xi


def blend(model_prob, dc_prob):
    """Dynamic blending: penalize large model-DC divergence."""
    gap = abs(model_prob - dc_prob)
    dc_w = 0.5 + (gap * 0.5)
    return (dc_prob * dc_w) + (model_prob * (1 - dc_w)), dc_w


def main():
    print("Loading models (Dixon-Coles + 3x XGBoost)...")
    dc, xgb_model, xgb_u35, xgb_u45, meta = load_models()
    features_o25 = meta["features"]
    features_u35 = meta["features_u35"]
    features_u45 = meta["features_u45"]
    # Superset for build_feature_row
    all_features = list(set(features_o25) | set(features_u35) | set(features_u45))

    print("Loading match data (for rolling lookups)...")
    df, team_view, df_standings = build_current_data(DB_PATH)

    teams = get_team_names(df)
    print(f"\nKnown teams ({len(teams)}):")
    for i, t in enumerate(teams, 1):
        print(f"  {i:>2}. {t}")

    print("\n" + "=" * 60)
    print("  MATCH PREDICTION (Dixon-Coles + XGBoost)")
    print("=" * 60)

    home_team = input("\nHome team: ").strip()
    away_team = input("Away team: ").strip()
    h_formation = input("Home formation (e.g. 4-2-3-1): ").strip()
    a_formation = input("Away formation (e.g. 3-4-2-1): ").strip()

    # Starting XI input
    current_season = SEASONS[-1]["label"]
    home_xi = _prompt_xi(home_team, current_season)
    away_xi = _prompt_xi(away_team, current_season)

    print("\nOdds (optional — improves accuracy):")
    b365_input = input("  Bet365 over 2.5 decimal odds (e.g. 1.85, or Enter to skip): ").strip()
    b365_over_odds = float(b365_input) if b365_input else None
    b365_u35_input = input("  Bet365 under 3.5 decimal odds (e.g. 1.45, or Enter to skip): ").strip()
    b365_under35_odds = float(b365_u35_input) if b365_u35_input else None
    b365_u45_input = input("  Bet365 under 4.5 decimal odds (e.g. 1.20, or Enter to skip): ").strip()
    b365_under45_odds = float(b365_u45_input) if b365_u45_input else None
    pin_input = input("  Pinnacle over 2.5 decimal odds (e.g. 1.92, or Enter to skip): ").strip()
    pinnacle_over_odds = float(pin_input) if pin_input else None

    # Build features (context + Dixon-Coles)
    X_full, dc_pred, xi_details, odds_sources = build_feature_row(
        home_team, away_team, h_formation, a_formation,
        df, team_view, df_standings, dc, all_features,
        b365_over_odds=b365_over_odds,
        pinnacle_over_odds=pinnacle_over_odds,
        b365_under35_odds=b365_under35_odds,
        b365_under45_odds=b365_under45_odds,
        home_xi=home_xi,
        away_xi=away_xi,
    )

    # Layer 2: XGBoost predictions (each model gets its own feature columns)
    xgb_prob = float(xgb_model.predict_proba(X_full[features_o25])[:, 1][0])
    xgb_u35_prob = float(xgb_u35.predict_proba(X_full[features_u35])[:, 1][0])
    xgb_u45_prob = float(xgb_u45.predict_proba(X_full[features_u45])[:, 1][0])

    # Weighted blending for all three lines
    blended_prob, dc_weight = blend(xgb_prob, dc_pred["prob_over25"])
    blended_u35, dc_w_u35 = blend(xgb_u35_prob, dc_pred["prob_under35"])
    blended_u45, dc_w_u45 = blend(xgb_u45_prob, dc_pred["prob_under45"])

    final_prob = blended_prob

    # Display results
    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)

    print(f"\n  {home_team} vs {away_team}")
    print(f"  Formations: {h_formation} vs {a_formation}")

    print(f"\n  --- Layer 1: Dixon-Coles ---")
    print(f"  Expected goals: {dc_pred['lambda_home']:.2f} - {dc_pred['lambda_away']:.2f}")
    print(f"  DC P(over 2.5): {dc_pred['prob_over25']:.1%}")
    print(f"  DC P(over 3.5): {dc_pred['prob_over35']:.1%}")
    print(f"  DC P(under 3.5): {dc_pred['prob_under35']:.1%}")
    print(f"  DC P(under 4.5): {dc_pred['prob_under45']:.1%}")

    # Top 5 most likely scorelines
    mat = dc_pred["scoreline_matrix"]
    scorelines = []
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            scorelines.append((x, y, mat[x, y]))
    scorelines.sort(key=lambda s: s[2], reverse=True)
    print(f"  Most likely scorelines:")
    for x, y, p in scorelines[:5]:
        print(f"    {x}-{y}: {p:.1%}")

    print(f"\n  --- Layer 2: Blended ---")
    print(f"  O/U 2.5: XGB={xgb_prob:.1%}  Blended={blended_prob:.1%}  (DC wt: {dc_weight:.0%})")
    print(f"  U 3.5:   XGB={xgb_u35_prob:.1%}  Blended={blended_u35:.1%}  (DC wt: {dc_w_u35:.0%})")
    print(f"  U 4.5:   XGB={xgb_u45_prob:.1%}  Blended={blended_u45:.1%}  (DC wt: {dc_w_u45:.0%})")
    # Odds debug
    print(f"\n  --- Odds Inputs ---")
    _fmt = lambda v: f"{v:.3f}" if v is not None else "NaN"
    _fmtv = lambda col: _fmt(float(X_full[col].iloc[0])) if col in X_full.columns and pd.notna(X_full[col].iloc[0]) else "NaN"
    print(f"  O/U 2.5: b365_prob_over25={_fmtv('b365_prob_over25')}  odds_dc_diff={_fmtv('odds_dc_diff')}  [{odds_sources['o25']}]")
    print(f"  U 3.5:   b365_prob_over35={_fmtv('b365_prob_over35')}  odds_dc_diff_35={_fmtv('odds_dc_diff_35')}  [{odds_sources['u35']}]")
    print(f"  U 4.5:   b365_prob_over45={_fmtv('b365_prob_over45')}  odds_dc_diff_45={_fmtv('odds_dc_diff_45')}  [{odds_sources['u45']}]")

    # XI match details
    for side, label in [("home", home_team), ("away", away_team)]:
        details = xi_details.get(side)
        if details:
            matched = [d for d in details if d["matched_name"]]
            print(f"\n  --- {label} XI Matching ({len(matched)}/{len(details)} matched) ---")
            for d in details:
                if d["matched_name"]:
                    print(f"    {d['input_name']:20s} -> {d['matched_name']:30s} "
                          f"xG/90={d['xg_per90']:.3f}  xA/90={d['xa_per90']:.3f}  ({d['status']})")
                else:
                    print(f"    {d['input_name']:20s} -> NOT FOUND ({d['status']})")

    print(f"\n  >>> Final P(over 2.5): {final_prob:.1%}")
    print(f"  >>> Prediction: {'OVER 2.5' if final_prob > 0.5 else 'UNDER 2.5'}")

    # Signal logic with under threshold
    poly_input = input("\n  Polymarket over 2.5 price (0-100, or Enter to skip): ").strip()
    if poly_input:
        poly_implied = float(poly_input) / 100.0
        edge = blended_prob - poly_implied
        print(f"\n  --- O/U 2.5 Edge Analysis ---")
        print(f"  Our blended:      {blended_prob:.1%}")
        print(f"  Polymarket:       {poly_implied:.1%}")
        print(f"  Edge:             {edge:+.1%}")

        if edge > 0.08 and blended_prob > 0.58:
            print(f"  Signal: BET OVER 2.5")
        elif edge < -0.08 and blended_prob < 0.58:
            print(f"  Signal: BET UNDER 2.5")
        elif edge < -0.08 and blended_prob >= 0.58:
            print(f"  Signal: PASS — Model still favors over — edge insufficient to bet against")
        else:
            print(f"  Signal: PASS — Edge below 8% threshold")

    # Under 3.5 / Under 4.5 lines (blended)
    print(f"\n  --- Additional Lines ---")
    print(f"  Blended P(under 3.5): {blended_u35:.1%}")
    print(f"  Blended P(under 4.5): {blended_u45:.1%}")

    poly_u35 = input("  Polymarket U3.5 price (0-100, or Enter to skip): ").strip()
    if poly_u35:
        u35_implied = float(poly_u35) / 100.0
        u35_edge = blended_u35 - u35_implied
        if u35_edge > 0.08 and blended_u35 > 0.62:
            signal_u35 = "BET U3.5"
        else:
            signal_u35 = "PASS"
        print(f"  U3.5: Blended={blended_u35:.1%}  Poly={u35_implied:.1%}  Edge={u35_edge:+.1%}  {signal_u35}")

    poly_u45 = input("  Polymarket U4.5 price (0-100, or Enter to skip): ").strip()
    if poly_u45:
        u45_implied = float(poly_u45) / 100.0
        u45_edge = blended_u45 - u45_implied
        if u45_edge > 0.08 and blended_u45 > 0.78:
            signal_u45 = "BET U4.5"
        else:
            signal_u45 = "PASS"
        print(f"  U4.5: Blended={blended_u45:.1%}  Poly={u45_implied:.1%}  Edge={u45_edge:+.1%}  {signal_u45}")


if __name__ == "__main__":
    main()
