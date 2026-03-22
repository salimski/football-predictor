"""
Streamlit dashboard for Bundesliga over/under 2.5 goals prediction.

Screen 1: Match Predictor (Dixon-Coles + XGBoost pipeline)
Screen 2: Model Tracker (log predictions, track ROI)

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import sys
import sqlite3
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

from predict import (
    load_models, build_current_data, build_feature_row,
    get_team_names, load_understat_lookups,
)
from features.player_features import _get_formation_xg, get_typical_xi
from config import DB_PATH, SEASONS

# ---------------------------------------------------------------------------
# Current 2025/26 Bundesliga teams
# ---------------------------------------------------------------------------

CURRENT_SEASON_TEAMS = {
    "1. FC Heidenheim", "1. FC Köln", "1899 Hoffenheim", "Bayer Leverkusen",
    "Bayern Munich", "Borussia Dortmund", "Borussia Mönchengladbach",
    "Eintracht Frankfurt", "FC Augsburg", "FC St. Pauli", "FSV Mainz 05",
    "Hamburger SV", "RB Leipzig", "SC Freiburg", "Union Berlin",
    "VfB Stuttgart", "VfL Wolfsburg", "Werder Bremen",
}

FORMATIONS = [
    "3-4-2-1", "3-4-3", "3-5-2", "4-2-3-1", "4-3-1-2", "4-3-2-1",
    "4-3-3", "4-4-1-1", "4-4-2", "4-5-1", "5-3-2", "5-4-1",
    "3-1-4-2", "3-4-1-2", "3-5-1-1", "4-1-3-2", "4-1-4-1", "4-2-2-2",
    "3-3-1-3", "3-3-3-1", "3-2-4-1",
]

TRACKER_FILE = "data/prediction_tracker.json"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    dc, xgb_model, xgb_u35, xgb_u45, meta = load_models()
    df, team_view, df_standings = build_current_data(DB_PATH)
    teams = get_team_names(df)
    formation_data, team_most_used, player_xg_lookup = load_understat_lookups(DB_PATH)
    return (dc, xgb_model, xgb_u35, xgb_u45, meta, df, team_view, df_standings, teams,
            formation_data, team_most_used)


def load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            text = f.read()
        # Replace bare NaN (invalid JSON) with null before parsing
        import re
        text = re.sub(r'\bNaN\b', 'null', text)
        return json.loads(text)
    return []


def _sanitize_for_json(records):
    """Replace NaN/inf with None so json.dump produces valid JSON."""
    import math
    clean = []
    for rec in records:
        row = {}
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
            else:
                row[k] = v
        clean.append(row)
    return clean


def save_tracker(records):
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    records = _sanitize_for_json(records)
    with open(TRACKER_FILE, "w") as f:
        json.dump(records, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Screen 1: Match Predictor
# ---------------------------------------------------------------------------

def _blend(model_prob, dc_prob):
    """Dynamic blending: penalize large model-DC divergence."""
    gap = abs(model_prob - dc_prob)
    dc_w = 0.5 + (gap * 0.5)
    return (dc_prob * dc_w) + (model_prob * (1 - dc_w)), dc_w


def screen_predictor():
    (dc, xgb_model, xgb_u35, xgb_u45, meta, df, team_view, df_standings, teams,
     formation_data, team_most_used) = load_pipeline()
    features_o25 = meta["features"]
    features_u35 = meta["features_u35"]
    features_u45 = meta["features_u45"]
    # Superset of all model feature lists for build_feature_row
    all_features = list(set(features_o25) | set(features_u35) | set(features_u45))

    # Sort current-season teams first, then historical teams
    current = sorted([t for t in teams if t in CURRENT_SEASON_TEAMS])
    historical = sorted([t for t in teams if t not in CURRENT_SEASON_TEAMS])
    dropdown_teams = current + historical

    current_season = SEASONS[-1]["label"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home")
        home_team = st.selectbox("Home team", dropdown_teams, key="home")
        h_formation = st.selectbox("Home formation", FORMATIONS, index=FORMATIONS.index("4-2-3-1"), key="hf")

    with col2:
        st.subheader("Away")
        away_team = st.selectbox("Away team", dropdown_teams, index=1, key="away")
        a_formation = st.selectbox("Away formation", FORMATIONS, index=FORMATIONS.index("4-2-3-1"), key="af")

    if home_team == away_team:
        st.warning("Home and away team must be different.")
        return

    # Starting XI inputs
    st.divider()
    st.markdown("**Starting XI**")
    xi_col1, xi_col2 = st.columns(2)

    with xi_col1:
        st.caption(f"{home_team}")
        if st.button("Load typical XI", key="load_h_xi"):
            typical = get_typical_xi(home_team, current_season)
            if typical:
                for i, name in enumerate(typical):
                    st.session_state[f"h_xi_{i}"] = name
            else:
                st.warning("No lineup data found.")
        h_xi_cols = st.columns(3)
        home_xi_inputs = []
        for i in range(11):
            with h_xi_cols[i % 3]:
                val = st.text_input(
                    f"Player {i+1}", value=st.session_state.get(f"h_xi_{i}", ""),
                    key=f"h_xi_input_{i}", label_visibility="collapsed",
                    placeholder=f"Player {i+1}",
                )
                home_xi_inputs.append(val.strip())

    with xi_col2:
        st.caption(f"{away_team}")
        if st.button("Load typical XI", key="load_a_xi"):
            typical = get_typical_xi(away_team, current_season)
            if typical:
                for i, name in enumerate(typical):
                    st.session_state[f"a_xi_{i}"] = name
            else:
                st.warning("No lineup data found.")
        a_xi_cols = st.columns(3)
        away_xi_inputs = []
        for i in range(11):
            with a_xi_cols[i % 3]:
                val = st.text_input(
                    f"Player {i+1}", value=st.session_state.get(f"a_xi_{i}", ""),
                    key=f"a_xi_input_{i}", label_visibility="collapsed",
                    placeholder=f"Player {i+1}",
                )
                away_xi_inputs.append(val.strip())

    # Build XI lists (only if at least 1 player entered)
    home_xi = [p for p in home_xi_inputs if p] or None
    away_xi = [p for p in away_xi_inputs if p] or None

    # Formation Intelligence section
    st.divider()
    st.markdown("**Formation Intelligence**")
    h_fxg = _get_formation_xg(current_season, home_team, h_formation,
                               formation_data, team_most_used)
    a_fxg = _get_formation_xg(current_season, away_team, a_formation,
                               formation_data, team_most_used)

    fi1, fi2, fi3 = st.columns(3)
    if h_fxg:
        fi1.metric(f"{home_team} xG/game in {h_formation}",
                   f"{h_fxg['xg_for_per_game']:.2f}")
    else:
        fi1.metric(f"{home_team} xG/game", "N/A")

    if a_fxg:
        fi2.metric(f"{away_team} xG/game in {a_formation}",
                   f"{a_fxg['xg_for_per_game']:.2f}")
    else:
        fi2.metric(f"{away_team} xG/game", "N/A")

    if h_fxg and a_fxg:
        matchup = h_fxg["xg_for_per_game"] + a_fxg["xg_for_per_game"]
        fi3.metric("Combined xG matchup", f"{matchup:.2f}")
    else:
        fi3.metric("Combined xG matchup", "N/A")

    st.divider()

    # Market inputs — B365 odds for all three lines
    st.markdown("**Bet365 Odds** (optional — improves accuracy)")
    b365_c1, b365_c2, b365_c3 = st.columns(3)
    with b365_c1:
        b365_odds_input = st.text_input(
            "Over 2.5 odds",
            value="",
            placeholder="e.g. 1.85",
            help="Decimal odds from Bet365 for over 2.5 goals.",
        )
    with b365_c2:
        b365_35_input = st.text_input(
            "Under 3.5 odds",
            value="",
            placeholder="e.g. 1.45",
            help="Decimal odds from Bet365 for under 3.5 goals. Converted to P(over 3.5) for the model.",
        )
    with b365_c3:
        b365_45_input = st.text_input(
            "Under 4.5 odds",
            value="",
            placeholder="e.g. 1.20",
            help="Decimal odds from Bet365 for under 4.5 goals. Converted to P(over 4.5) for the model.",
        )

    st.markdown("**Polymarket Prices**")
    poly_c1, poly_c2, poly_c3 = st.columns(3)
    with poly_c1:
        poly_price = st.number_input(
            "O2.5 price (0-100)",
            min_value=0, max_value=100, value=0, step=1,
            help="Polymarket price for over 2.5 goals (0 to skip)",
        )
    with poly_c2:
        poly_u35 = st.number_input(
            "U3.5 price (0-100)",
            min_value=0, max_value=100, value=0, step=1,
            help="Polymarket price for under 3.5 goals (0 to skip)",
        )
    with poly_c3:
        poly_u45 = st.number_input(
            "U4.5 price (0-100)",
            min_value=0, max_value=100, value=0, step=1,
            help="Polymarket price for under 4.5 goals (0 to skip)",
        )

    # Parse B365 odds (all three lines)
    def _parse_odds(raw, label):
        if not raw.strip():
            return None
        try:
            val = float(raw.strip())
            if val <= 1.0:
                st.error(f"{label}: decimal odds must be > 1.0")
                return None
            return val
        except ValueError:
            st.error(f"{label}: invalid format — enter a decimal like 1.85")
            return None

    b365_over_odds = _parse_odds(b365_odds_input, "Over 2.5")
    b365_under35_odds = _parse_odds(b365_35_input, "Under 3.5")
    b365_under45_odds = _parse_odds(b365_45_input, "Under 4.5")

    if st.button("Predict", type="primary"):
        with st.spinner("Running prediction pipeline..."):
            X_full, dc_pred, xi_details, odds_sources = build_feature_row(
                home_team, away_team, h_formation, a_formation,
                df, team_view, df_standings, dc, all_features,
                b365_over_odds=b365_over_odds,
                b365_under35_odds=b365_under35_odds,
                b365_under45_odds=b365_under45_odds,
                home_xi=home_xi,
                away_xi=away_xi,
            )
            xgb_prob = float(xgb_model.predict_proba(X_full[features_o25])[:, 1][0])
            xgb_u35_prob = float(xgb_u35.predict_proba(X_full[features_u35])[:, 1][0])
            xgb_u45_prob = float(xgb_u45.predict_proba(X_full[features_u45])[:, 1][0])

            # Weighted blending for all three lines
            blended_prob, dc_weight = _blend(xgb_prob, dc_pred["prob_over25"])
            blended_u35, dc_w_u35 = _blend(xgb_u35_prob, dc_pred["prob_under35"])
            blended_u45, dc_w_u45 = _blend(xgb_u45_prob, dc_pred["prob_under45"])

            final_prob = blended_prob

        b365_implied = (1.0 / b365_over_odds) if b365_over_odds else None

        # Store results in session state so they persist across reruns
        st.session_state["last_prediction"] = {
            "home_team": home_team,
            "away_team": away_team,
            "h_formation": h_formation,
            "a_formation": a_formation,
            "dc_pred": dc_pred,
            "xgb_prob": xgb_prob,
            "blended_prob": blended_prob,
            "final_prob": final_prob,
            "dc_weight": dc_weight,
            "blended_u35": blended_u35,
            "dc_w_u35": dc_w_u35,
            "blended_u45": blended_u45,
            "dc_w_u45": dc_w_u45,
            "b365_implied": b365_implied,
            "poly_price": poly_price,
            "poly_u35": poly_u35,
            "poly_u45": poly_u45,
            "xi_details": xi_details,
            "odds_sources": odds_sources,
        }

    # ── Display results (from session state, survives reruns) ─────────
    if "last_prediction" not in st.session_state:
        return

    pred = st.session_state["last_prediction"]
    dc_pred = pred["dc_pred"]
    final_prob = pred["final_prob"]
    xgb_prob = pred["xgb_prob"]
    blended_prob = pred.get("blended_prob", final_prob)
    dc_weight = pred.get("dc_weight", 0.5)
    b365_implied = pred["b365_implied"]
    poly_price = pred["poly_price"]
    poly_u35 = pred.get("poly_u35", 0)
    poly_u45 = pred.get("poly_u45", 0)
    blended_u35 = pred.get("blended_u35", dc_pred.get("prob_under35", 0))
    blended_u45 = pred.get("blended_u45", dc_pred.get("prob_under45", 0))
    odds_sources = pred.get("odds_sources", {"o25": "DC only", "u35": "DC only", "u45": "DC only"})

    st.divider()
    st.subheader(f"{pred['home_team']} vs {pred['away_team']}")

    # Overconfidence warning
    if blended_prob > 0.80:
        st.warning(
            f"Blended probability is {blended_prob:.1%}. "
            f"Calibration data shows effective probability is ~{blended_prob - 0.10:.1%} "
            f"in this range. Adjust expectations accordingly."
        )

    # Four key probabilities side by side
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("DC P(over 2.5)", f"{dc_pred['prob_over25']:.1%}")
    m2.metric("Model P(over 2.5)", f"{xgb_prob:.1%}")
    m3.metric("Blended P(over 2.5)", f"{blended_prob:.1%}")
    m3.caption(f"DC weight: {dc_weight:.0%}")
    if b365_implied is not None:
        m4.metric("B365 implied", f"{b365_implied:.1%}")
    else:
        m4.metric("B365 implied", "-")

    # DC details
    st.markdown("**Dixon-Coles**")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("lambda home", f"{dc_pred['lambda_home']:.2f}")
    d2.metric("lambda away", f"{dc_pred['lambda_away']:.2f}")
    d3.metric("P(under 2.5)", f"{dc_pred['prob_under25']:.1%}")
    d4.metric("P(under 3.5)", f"{dc_pred['prob_under35']:.1%}")
    d5.metric("P(under 4.5)", f"{dc_pred.get('prob_under45', 0):.1%}")

    # XI matching details
    xi_details = pred.get("xi_details", {})
    for side, label in [("home", pred["home_team"]), ("away", pred["away_team"])]:
        details = xi_details.get(side) if xi_details else None
        if details:
            matched = [d for d in details if d["matched_name"]]
            with st.expander(f"{label} XI Matching ({len(matched)}/{len(details)} found)"):
                rows_data = []
                for d in details:
                    rows_data.append({
                        "Input": d["input_name"],
                        "Matched": d["matched_name"] or "-",
                        "xG/90": f"{d['xg_per90']:.3f}" if d["xg_per90"] is not None else "-",
                        "xA/90": f"{d['xa_per90']:.3f}" if d["xa_per90"] is not None else "-",
                        "Status": d["status"],
                    })
                st.table(pd.DataFrame(rows_data))

    # ── Scoreline heatmap ─────────────────────────────────────────
    st.markdown("**Scoreline Probabilities**")
    mat = dc_pred["scoreline_matrix"]
    display_n = 8
    mat_display = mat[:display_n, :display_n]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(mat_display, cmap="YlOrRd", origin="upper", vmin=0)

    for i in range(display_n):
        for j in range(display_n):
            val = mat_display[i, j]
            color = "white" if val > mat_display.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(display_n))
    ax.set_yticks(range(display_n))
    ax.set_xticklabels(range(display_n))
    ax.set_yticklabels(range(display_n))
    ax.set_xlabel(f"{pred['away_team']} goals")
    ax.set_ylabel(f"{pred['home_team']} goals")
    ax.set_title("P(home goals, away goals)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Top scorelines
    scorelines = []
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            scorelines.append((x, y, mat[x, y]))
    scorelines.sort(key=lambda s: s[2], reverse=True)
    top5 = ", ".join(f"{x}-{y} ({p:.1%})" for x, y, p in scorelines[:5])
    st.caption(f"Most likely: {top5}")

    # ── Lines Panel ─────────────────────────────────────────────
    has_any_poly = poly_price > 0 or poly_u35 > 0 or poly_u45 > 0
    if has_any_poly:
        st.divider()
        st.markdown("**Lines Panel — Polymarket Edge Analysis**")

        # --- O/U 2.5 line ---
        signal_25 = "PASS"
        edge_25 = None
        if poly_price > 0:
            poly_implied = poly_price / 100.0
            edge_25 = blended_prob - poly_implied

            if edge_25 > 0.08 and blended_prob > 0.58:
                if blended_prob > 0.80:
                    adj_edge = (blended_prob - 0.10) - poly_implied
                    if adj_edge > 0.08:
                        signal_25 = "BET OVER (caution)"
                    else:
                        signal_25 = "PASS"
                else:
                    signal_25 = "BET OVER"
            elif edge_25 < -0.08 and blended_prob < 0.58:
                signal_25 = "BET UNDER"
            elif edge_25 < -0.08 and blended_prob >= 0.58:
                signal_25 = "PASS"
            else:
                signal_25 = "PASS"

        # --- U3.5 line (blended) ---
        signal_u35 = "PASS"
        edge_u35 = None
        dc_u35 = dc_pred.get("prob_under35", 0)
        if poly_u35 > 0:
            u35_implied = poly_u35 / 100.0
            edge_u35 = blended_u35 - u35_implied
            if edge_u35 > 0.08 and blended_u35 > 0.62:
                signal_u35 = "BET U3.5"
            else:
                signal_u35 = "PASS"

        # --- U4.5 line (blended) ---
        signal_u45 = "PASS"
        edge_u45 = None
        dc_u45 = dc_pred.get("prob_under45", 0)
        if poly_u45 > 0:
            u45_implied = poly_u45 / 100.0
            edge_u45 = blended_u45 - u45_implied
            if edge_u45 > 0.08 and blended_u45 > 0.78:
                signal_u45 = "BET U4.5"
            else:
                signal_u45 = "PASS"

        # Build table
        lines_data = []
        if poly_price > 0:
            lines_data.append({
                "Line": "O/U 2.5",
                "DC Prob": f"{dc_pred['prob_over25']:.1%}",
                "Model/Blended": f"{blended_prob:.1%}",
                "Polymarket": f"{poly_price}¢",
                "Edge": f"{edge_25:+.1%}" if edge_25 is not None else "-",
                "Signal": signal_25,
                "Odds Source": odds_sources["o25"],
            })
        if poly_u35 > 0:
            lines_data.append({
                "Line": "U 3.5",
                "DC Prob": f"{dc_u35:.1%}",
                "Model/Blended": f"{blended_u35:.1%}",
                "Polymarket": f"{poly_u35}¢",
                "Edge": f"{edge_u35:+.1%}" if edge_u35 is not None else "-",
                "Signal": signal_u35,
                "Odds Source": odds_sources["u35"],
            })
        if poly_u45 > 0:
            lines_data.append({
                "Line": "U 4.5",
                "DC Prob": f"{dc_u45:.1%}",
                "Model/Blended": f"{blended_u45:.1%}",
                "Polymarket": f"{poly_u45}¢",
                "Edge": f"{edge_u45:+.1%}" if edge_u45 is not None else "-",
                "Signal": signal_u45,
                "Odds Source": odds_sources["u45"],
            })

        st.table(pd.DataFrame(lines_data))

        # Signal callouts
        for row in lines_data:
            if "BET" in row["Signal"]:
                st.success(f"{row['Line']}: {row['Signal']} — Edge: {row['Edge']}")

        # Note about under threshold
        if poly_price > 0 and edge_25 is not None and edge_25 < -0.08 and blended_prob >= 0.58:
            st.info("O/U 2.5: Model still favors over — edge insufficient to bet against")

        if not any("BET" in r["Signal"] for r in lines_data):
            best_signal = signal_25 if poly_price > 0 else "PASS"
            if best_signal == "PASS":
                st.info("No actionable edges found across all lines.")

        st.caption("All three lines use DC + XGBoost blended probabilities. U3.5 threshold: blended > 62%. U4.5 threshold: blended > 78%.")

        # Quick-log button
        st.divider()
        if st.button("Log this prediction to tracker"):
            record = {
                "date": str(date.today()),
                "match": f"{pred['home_team']} vs {pred['away_team']}",
                "home_formation": pred["h_formation"],
                "away_formation": pred["a_formation"],
                "our_prob": round(final_prob * 100, 1),
                "dc_prob": round(dc_pred["prob_over25"] * 100, 1),
                "blended_prob": round(blended_prob * 100, 1),
                "b365_implied": round(b365_implied * 100, 1) if b365_implied else None,
                "poly_price": poly_price,
                "edge": round(edge_25 * 100, 1) if edge_25 is not None else None,
                "signal": signal_25,
                "bet_placed": False,
                "result": None,
                # U3.5 fields
                "dc_prob_u35": round(dc_u35 * 100, 1),
                "blended_u35": round(blended_u35 * 100, 1),
                "poly_u35": poly_u35 if poly_u35 > 0 else None,
                "edge_u35": round(edge_u35 * 100, 1) if edge_u35 is not None else None,
                "signal_u35": signal_u35 if poly_u35 > 0 else None,
                "bet_placed_u35": False,
                "result_u35": None,
                # U4.5 fields
                "dc_prob_u45": round(dc_u45 * 100, 1),
                "blended_u45": round(blended_u45 * 100, 1),
                "poly_u45": poly_u45 if poly_u45 > 0 else None,
                "edge_u45": round(edge_u45 * 100, 1) if edge_u45 is not None else None,
                "signal_u45": signal_u45 if poly_u45 > 0 else None,
                "bet_placed_u45": False,
                "result_u45": None,
            }
            tracker = load_tracker()
            tracker.append(record)
            save_tracker(tracker)
            st.success("Logged to tracker.")


# ---------------------------------------------------------------------------
# Screen 2: Model Tracker
# ---------------------------------------------------------------------------

def screen_tracker():
    tracker = load_tracker()

    if not tracker:
        st.info("No predictions logged yet. Use the Match Predictor to log predictions.")

    # ── Add manual prediction ─────────────────────────────────────────
    with st.expander("Add prediction manually"):
        with st.form("add_prediction_form", clear_on_submit=True):
            ac1, ac2 = st.columns(2)
            add_date = ac1.date_input("Date", value=date.today())
            add_match = ac2.text_input("Match", placeholder="Home Team vs Away Team")

            ac3, ac4, ac5 = st.columns(3)
            add_blended = ac3.number_input("Blended P(O2.5) %", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            add_dc = ac4.number_input("DC P(O2.5) %", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            add_poly = ac5.number_input("Poly Price (O2.5)", min_value=0, max_value=100, value=50, step=1)

            ac6, ac7, ac8 = st.columns(3)
            add_b365 = ac6.number_input("B365 Implied %", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            add_signal = ac7.selectbox("Signal", ["PASS", "BET OVER", "BET UNDER"])
            add_result = ac8.selectbox("Result", [None, "over", "under"])

            submitted = st.form_submit_button("Add to tracker")
            if submitted and add_match.strip():
                edge_val = round(add_blended - add_poly, 1) if add_poly > 0 else None
                new_record = {
                    "date": str(add_date),
                    "match": add_match.strip(),
                    "home_formation": None,
                    "away_formation": None,
                    "our_prob": add_blended,
                    "dc_prob": add_dc,
                    "blended_prob": add_blended,
                    "b365_implied": add_b365,
                    "poly_price": add_poly,
                    "edge": edge_val,
                    "signal": add_signal,
                    "bet_placed": "BET" in add_signal,
                    "result": add_result,
                    "dc_prob_u35": None, "blended_u35": None, "poly_u35": None,
                    "edge_u35": None, "signal_u35": None, "bet_placed_u35": False, "result_u35": None,
                    "dc_prob_u45": None, "blended_u45": None, "poly_u45": None,
                    "edge_u45": None, "signal_u45": None, "bet_placed_u45": False, "result_u45": None,
                }
                tracker.append(new_record)
                save_tracker(tracker)
                st.success(f"Added: {add_match.strip()}")
                st.rerun()

    if not tracker:
        return

    df = pd.DataFrame(tracker)

    # Editable table
    st.subheader("Prediction Log")
    st.caption("Edit 'bet_placed' and 'result' columns below, then click Save.")

    # Ensure new columns exist for older records
    for col, default in [
        ("blended_prob", None),
        ("dc_prob_u35", None), ("blended_u35", None), ("poly_u35", None),
        ("edge_u35", None), ("signal_u35", None), ("bet_placed_u35", False),
        ("result_u35", None),
        ("dc_prob_u45", None), ("blended_u45", None), ("poly_u45", None),
        ("edge_u45", None), ("signal_u45", None), ("bet_placed_u45", False),
        ("result_u45", None),
    ]:
        if col not in df.columns:
            df[col] = default

    # Add row index for selection
    df.insert(0, "#", range(1, len(df) + 1))

    # O/U 2.5 tab and U3.5/U4.5 tabs
    t_25, t_35, t_45 = st.tabs(["O/U 2.5", "Under 3.5", "Under 4.5"])

    with t_25:
        edited = st.data_editor(
            df,
            column_config={
                "#": st.column_config.NumberColumn("#", disabled=True, width="small"),
                "date": st.column_config.TextColumn("Date", disabled=True),
                "match": st.column_config.TextColumn("Match", disabled=True),
                "home_formation": None,
                "away_formation": None,
                "our_prob": st.column_config.NumberColumn("XGB P(O2.5) %", format="%.1f", disabled=True),
                "dc_prob": st.column_config.NumberColumn("DC P(O2.5) %", format="%.1f", disabled=True),
                "blended_prob": st.column_config.NumberColumn("Blended %", format="%.1f", disabled=True),
                "b365_implied": st.column_config.NumberColumn("B365 %", format="%.1f", disabled=True),
                "poly_price": st.column_config.NumberColumn("Poly Price", format="%.0f", disabled=True),
                "edge": st.column_config.NumberColumn("Edge %", format="%+.1f", disabled=True),
                "signal": st.column_config.TextColumn("Signal", disabled=True),
                "bet_placed": st.column_config.CheckboxColumn("Bet?"),
                "result": st.column_config.SelectboxColumn("Result", options=["over", "under", None]),
                # Hide U3.5/U4.5 columns
                "dc_prob_u35": None, "blended_u35": None, "poly_u35": None,
                "edge_u35": None, "signal_u35": None, "bet_placed_u35": None,
                "result_u35": None,
                "dc_prob_u45": None, "blended_u45": None, "poly_u45": None,
                "edge_u45": None, "signal_u45": None, "bet_placed_u45": None,
                "result_u45": None,
            },
            hide_index=True,
            use_container_width=True,
            key="tracker_editor_25",
        )

    with t_35:
        df_u35 = df[df["poly_u35"].notna() & (df["poly_u35"] != 0)].copy() if "poly_u35" in df.columns else pd.DataFrame()
        if len(df_u35) > 0:
            edited_u35 = st.data_editor(
                df_u35[["#", "date", "match", "dc_prob_u35", "blended_u35", "poly_u35", "edge_u35", "signal_u35", "bet_placed_u35", "result_u35"]],
                column_config={
                    "#": st.column_config.NumberColumn("#", disabled=True, width="small"),
                    "date": st.column_config.TextColumn("Date", disabled=True),
                    "match": st.column_config.TextColumn("Match", disabled=True),
                    "dc_prob_u35": st.column_config.NumberColumn("DC P(U3.5) %", format="%.1f", disabled=True),
                    "blended_u35": st.column_config.NumberColumn("Blended %", format="%.1f", disabled=True),
                    "poly_u35": st.column_config.NumberColumn("Poly U3.5", format="%.0f", disabled=True),
                    "edge_u35": st.column_config.NumberColumn("Edge %", format="%+.1f", disabled=True),
                    "signal_u35": st.column_config.TextColumn("Signal", disabled=True),
                    "bet_placed_u35": st.column_config.CheckboxColumn("Bet?"),
                    "result_u35": st.column_config.SelectboxColumn("Result", options=["under", "over", None]),
                },
                hide_index=True,
                use_container_width=True,
                key="tracker_editor_u35",
            )
            # Merge U3.5 edits back into main df using row number
            for i, (orig_idx, _) in enumerate(df_u35.iterrows()):
                if i < len(edited_u35):
                    edited.loc[orig_idx, "bet_placed_u35"] = edited_u35.iloc[i]["bet_placed_u35"]
                    edited.loc[orig_idx, "result_u35"] = edited_u35.iloc[i]["result_u35"]
        else:
            st.info("No U3.5 predictions logged yet.")

    with t_45:
        df_u45 = df[df["poly_u45"].notna() & (df["poly_u45"] != 0)].copy() if "poly_u45" in df.columns else pd.DataFrame()
        if len(df_u45) > 0:
            edited_u45 = st.data_editor(
                df_u45[["#", "date", "match", "dc_prob_u45", "blended_u45", "poly_u45", "edge_u45", "signal_u45", "bet_placed_u45", "result_u45"]],
                column_config={
                    "#": st.column_config.NumberColumn("#", disabled=True, width="small"),
                    "date": st.column_config.TextColumn("Date", disabled=True),
                    "match": st.column_config.TextColumn("Match", disabled=True),
                    "dc_prob_u45": st.column_config.NumberColumn("DC P(U4.5) %", format="%.1f", disabled=True),
                    "blended_u45": st.column_config.NumberColumn("Blended %", format="%.1f", disabled=True),
                    "poly_u45": st.column_config.NumberColumn("Poly U4.5", format="%.0f", disabled=True),
                    "edge_u45": st.column_config.NumberColumn("Edge %", format="%+.1f", disabled=True),
                    "signal_u45": st.column_config.TextColumn("Signal", disabled=True),
                    "bet_placed_u45": st.column_config.CheckboxColumn("Bet?"),
                    "result_u45": st.column_config.SelectboxColumn("Result", options=["under", "over", None]),
                },
                hide_index=True,
                use_container_width=True,
                key="tracker_editor_u45",
            )
            for i, (orig_idx, _) in enumerate(df_u45.iterrows()):
                if i < len(edited_u45):
                    edited.loc[orig_idx, "bet_placed_u45"] = edited_u45.iloc[i]["bet_placed_u45"]
                    edited.loc[orig_idx, "result_u45"] = edited_u45.iloc[i]["result_u45"]
        else:
            st.info("No U4.5 predictions logged yet.")

    # ── Save & Delete buttons ─────────────────────────────────────────
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        if st.button("Save changes", type="primary"):
            save_df = edited.drop(columns=["#"], errors="ignore")
            save_tracker(save_df.to_dict(orient="records"))
            st.success("Saved.")
            st.rerun()

    # Delete by row number
    with btn_col2:
        with st.popover("Delete predictions"):
            st.caption("Enter row numbers to delete (shown in '#' column).")
            del_input = st.text_input("Row numbers (comma-separated)", placeholder="e.g. 1, 3, 5", key="del_rows_input")
            if st.button("Confirm delete", type="secondary"):
                if del_input.strip():
                    try:
                        to_delete = {int(x.strip()) for x in del_input.split(",")}
                        # Row numbers are 1-based
                        remaining = [rec for i, rec in enumerate(tracker) if (i + 1) not in to_delete]
                        deleted_count = len(tracker) - len(remaining)
                        if deleted_count > 0:
                            save_tracker(remaining)
                            st.success(f"Deleted {deleted_count} prediction(s).")
                            st.rerun()
                        else:
                            st.warning("No matching row numbers found.")
                    except ValueError:
                        st.error("Invalid input. Use comma-separated numbers (e.g. 1, 3, 5).")

    # ── Running stats ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Running Stats")

    resolved = edited[edited["result"].notna() & (edited["result"] != "")]
    bets = edited[edited["bet_placed"] == True]
    resolved_bets = bets[bets["result"].notna() & (bets["result"] != "")]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total predictions", len(df))
    s2.metric("Bets placed", len(bets))

    # All predictions accuracy
    if len(resolved) > 0:
        correct = 0
        for _, row in resolved.iterrows():
            # Use blended_prob if available, fall back to our_prob
            prob = row.get("blended_prob")
            if pd.isna(prob):
                prob = row["our_prob"]
            predicted_over = prob > 50
            actual_over = row["result"] == "over"
            if predicted_over == actual_over:
                correct += 1
        pred_accuracy = correct / len(resolved)
        s3.metric("Prediction hit rate", f"{pred_accuracy:.0%}", help="All resolved predictions")
    else:
        s3.metric("Prediction hit rate", "-")

    # Bet-specific stats
    if len(resolved_bets) > 0:
        wins = 0
        total_edge = 0.0
        total_roi = 0.0

        for _, row in resolved_bets.iterrows():
            signal = row.get("signal", "")
            actual = row["result"]
            edge_pct = abs(row["edge"]) if pd.notna(row["edge"]) else 0
            poly = row["poly_price"] / 100.0 if pd.notna(row["poly_price"]) else 0

            if "OVER" in str(signal):
                won = actual == "over"
            elif "UNDER" in str(signal):
                won = actual == "under"
            else:
                continue

            if won:
                wins += 1
                if "OVER" in str(signal):
                    payout = (1.0 / poly) - 1.0 if poly > 0 else 0
                else:
                    payout = (1.0 / (1.0 - poly)) - 1.0 if poly < 1 else 0
                total_roi += payout
            else:
                total_roi -= 1.0

            total_edge += edge_pct

        bet_hit_rate = wins / len(resolved_bets) if len(resolved_bets) > 0 else 0
        avg_edge = total_edge / len(resolved_bets) if len(resolved_bets) > 0 else 0
        roi_pct = (total_roi / len(resolved_bets)) * 100 if len(resolved_bets) > 0 else 0

        s4.metric("Bet hit rate", f"{bet_hit_rate:.0%}")

        b1, b2, b3 = st.columns(3)
        b1.metric("Avg edge", f"{avg_edge:.1f}%")
        b2.metric("Est. ROI", f"{roi_pct:+.1f}%", help="Per unit bet, resolved bets only")
        b3.metric("Resolved bets", f"{len(resolved_bets)}")

        # Win/loss breakdown
        st.caption(f"Wins: {wins} / Losses: {len(resolved_bets) - wins}")
    else:
        s4.metric("Bet hit rate", "-")
        st.caption("No resolved bets yet. Mark bets as placed and enter results to see stats.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Bundesliga Goals Predictor", layout="wide")
st.title("Bundesliga Goals Predictor")

tab1, tab2 = st.tabs(["Match Predictor", "Model Tracker"])

with tab1:
    screen_predictor()

with tab2:
    screen_tracker()
