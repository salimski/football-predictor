"""
Diagnostic analysis: why context features aren't helping XGBoost.

1. Point-biserial correlation of every non-DC feature with target_over25
2. Rolling window variants (5, 10, weighted-5)
3. Feature group ablation
"""

import sqlite3
import os
import sys
import json

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH
from model.dixon_coles import DixonColesModel
from model.train import CONTEXT_FEATURES, DC_FEATURES, TARGET_COL, load_features, add_dc_features


# =====================================================================
# Shared setup
# =====================================================================

def setup():
    """Load data, fit DC, return train/test with DC features attached."""
    df = load_features(DB_PATH)
    train_df = df[df["season"].isin(["2021/22", "2022/23", "2023/24"])].copy()
    test_df = df[df["season"].isin(["2024/25", "2025/26"])].copy()

    dc = DixonColesModel(xi=0.0065)
    dc.fit(
        home_teams=train_df["home_team"].tolist(),
        away_teams=train_df["away_team"].tolist(),
        goals_home=train_df["goals_home"].tolist(),
        goals_away=train_df["goals_away"].tolist(),
        match_dates=train_df["match_date"].tolist(),
    )
    train_df = add_dc_features(train_df, dc)
    test_df = add_dc_features(test_df, dc)
    return train_df, test_df, dc


def train_and_eval(X_train, y_train, X_test, y_test):
    """Train XGBoost with standard params, return AUC."""
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


# =====================================================================
# 1. Point-biserial correlation
# =====================================================================

def analysis_correlation(train_df):
    print("=" * 60)
    print("  1. POINT-BISERIAL CORRELATION WITH target_over25")
    print("=" * 60)

    target = train_df[TARGET_COL]
    results = []

    # All context features (non-DC)
    for col in CONTEXT_FEATURES:
        if col not in train_df.columns:
            continue
        series = train_df[col]
        valid = series.notna() & target.notna()
        if valid.sum() < 30:
            results.append((col, np.nan, 0))
            continue
        r, p = pointbiserialr(target[valid], series[valid])
        results.append((col, r, p))

    # Also check DC features for reference
    for col in DC_FEATURES:
        if col not in train_df.columns:
            continue
        series = train_df[col]
        valid = series.notna() & target.notna()
        if valid.sum() < 30:
            continue
        r, p = pointbiserialr(target[valid], series[valid])
        results.append((col, r, p))

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)

    print(f"\n  {'Feature':<35} {'Corr':>7} {'p-value':>10} {'Sig':>4}")
    print(f"  {'-'*60}")

    # Top 20
    print("\n  --- Top 20 by |correlation| ---")
    for col, r, p in results[:20]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<35} {r:>+7.4f} {p:>10.2e} {sig:>4}")

    # Bottom 10
    print("\n  --- Bottom 10 (weakest) ---")
    # Filter out NaN
    valid_results = [r for r in results if not np.isnan(r[1])]
    for col, r, p in valid_results[-10:]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<35} {r:>+7.4f} {p:>10.2e} {sig:>4}")

    return results


# =====================================================================
# 2. Rolling window experiment
# =====================================================================

def analysis_rolling_windows(train_df, test_df):
    print("\n" + "=" * 60)
    print("  2. ROLLING WINDOW EXPERIMENT")
    print("=" * 60)

    from features.builder import build_team_view, ROLL_COLS, VENUE_ROLL_COLS

    # We need the raw match data to rebuild rolling features
    full_df = load_features(DB_PATH)

    # Build team_view from the features table's post-match data
    # We need: match_id, match_date, team, venue, goals_scored, goals_conceded,
    #          total_goals, xg_for, xg_against, shots_on_target, shots_total,
    #          corners, possession
    rows = []
    for _, r in full_df.iterrows():
        gh = r["goals_home"]
        ga = r["goals_away"]
        tg = gh + ga if pd.notna(gh) and pd.notna(ga) else None
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

    def compute_rolling_custom(team, before_date, team_view, window, weighted=False):
        """Custom rolling with variable window and optional linear decay."""
        prior = team_view[
            (team_view["team"] == team) &
            (team_view["match_date"] < before_date)
        ].sort_values("match_date").tail(window)

        n = len(prior)
        if n == 0:
            return {col: (0 if col == "games_available" else None) for col in ROLL_COLS}

        if weighted and n > 1:
            # Linear decay: most recent = 1.0, oldest = 1.0 - 0.1*(n-1)
            # For n=5: [0.6, 0.7, 0.8, 0.9, 1.0]
            w = np.linspace(1.0 - 0.1 * (n - 1), 1.0, n)
            w = w / w.sum()  # normalize
        else:
            w = np.ones(n) / n

        def _wmean(col):
            s = prior[col].values
            mask = ~pd.isna(s)
            if not mask.any():
                return None
            return float(np.average(s[mask].astype(float), weights=w[mask]))

        mask = prior["goals_scored"].notna() & prior["goals_conceded"].notna()
        if mask.any():
            scored = prior.loc[mask, "goals_scored"].values.astype(float)
            conceded = prior.loc[mask, "goals_conceded"].values.astype(float)
            w_valid = w[mask.values]
            w_norm = w_valid / w_valid.sum()
            wins = float(np.sum(w_norm * (scored > conceded)))
            draws = float(np.sum(w_norm * (scored == conceded)))
        else:
            wins = draws = 0.0

        tg = prior["total_goals"].values
        tg_mask = ~pd.isna(tg)
        if tg_mask.any():
            over25_rate = float(np.average(tg[tg_mask].astype(float) > 2.5, weights=w[tg_mask]))
        else:
            over25_rate = None

        return {
            "games_available": n,
            "goals_scored": _wmean("goals_scored"),
            "goals_conceded": _wmean("goals_conceded"),
            "total_goals": _wmean("total_goals"),
            "over25_rate": over25_rate,
            "xg_for": _wmean("xg_for"),
            "xg_against": _wmean("xg_against"),
            "shots_on_target": _wmean("shots_on_target"),
            "shots_total": _wmean("shots_total"),
            "corners": _wmean("corners"),
            "possession": _wmean("possession"),
            "wins": wins,
            "draws": draws,
        }

    def compute_venue_custom(team, venue, before_date, team_view, window, weighted=False):
        prior = team_view[
            (team_view["team"] == team) &
            (team_view["venue"] == venue) &
            (team_view["match_date"] < before_date)
        ].sort_values("match_date").tail(window)
        n = len(prior)
        if n == 0:
            return {"venue_goals_scored": None, "venue_goals_conceded": None}
        if weighted and n > 1:
            w = np.linspace(1.0 - 0.1 * (n - 1), 1.0, n)
            w = w / w.sum()
            gs = prior["goals_scored"].values.astype(float)
            gc = prior["goals_conceded"].values.astype(float)
            return {
                "venue_goals_scored": float(np.average(gs[~np.isnan(gs)], weights=w[~np.isnan(gs)])),
                "venue_goals_conceded": float(np.average(gc[~np.isnan(gc)], weights=w[~np.isnan(gc)])),
            }
        return {
            "venue_goals_scored": float(prior["goals_scored"].mean(skipna=True)),
            "venue_goals_conceded": float(prior["goals_conceded"].mean(skipna=True)),
        }

    def rebuild_rolling(df_subset, window, weighted=False):
        """Recompute all rolling features for a DataFrame subset."""
        out = df_subset.copy()
        roll_feature_cols = [c for c in ROLL_COLS if c != "games_available"]

        h_rolls = []
        a_rolls = []
        h_venues = []
        a_venues = []

        for _, row in out.iterrows():
            bd = pd.Timestamp(row["match_date"])
            h_rolls.append(compute_rolling_custom(row["home_team"], bd, tv, window, weighted))
            a_rolls.append(compute_rolling_custom(row["away_team"], bd, tv, window, weighted))
            h_venues.append(compute_venue_custom(row["home_team"], "home", bd, tv, window, weighted))
            a_venues.append(compute_venue_custom(row["away_team"], "away", bd, tv, window, weighted))

        for col in roll_feature_cols:
            out[f"h_roll_{col}"] = [r[col] for r in h_rolls]
            out[f"a_roll_{col}"] = [r[col] for r in a_rolls]

        for col in ["venue_goals_scored", "venue_goals_conceded"]:
            out[f"h_{col}"] = [r[col] for r in h_venues]
            out[f"a_{col}"] = [r[col] for r in a_venues]

        return out

    # DC model (reuse from setup)
    dc = DixonColesModel(xi=0.0065)
    dc.fit(
        home_teams=train_df["home_team"].tolist(),
        away_teams=train_df["away_team"].tolist(),
        goals_home=train_df["goals_home"].tolist(),
        goals_away=train_df["goals_away"].tolist(),
        match_dates=train_df["match_date"].tolist(),
    )

    all_features = CONTEXT_FEATURES + DC_FEATURES
    available = [c for c in all_features if c in train_df.columns]
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    # Baseline: current (window=5, unweighted) — already in train_df/test_df
    auc_base = train_and_eval(train_df[available], y_train, test_df[available], y_test)
    print(f"\n  Current (window=5, unweighted):  AUC = {auc_base:.4f}")

    # Variant A: window=10
    print("\n  Rebuilding rolling features with window=10...")
    train_10 = rebuild_rolling(train_df, window=10, weighted=False)
    test_10 = rebuild_rolling(test_df, window=10, weighted=False)
    train_10 = add_dc_features(train_10, dc)
    test_10 = add_dc_features(test_10, dc)
    avail_10 = [c for c in available if c in train_10.columns]
    auc_10 = train_and_eval(train_10[avail_10], y_train, test_10[avail_10], y_test)
    print(f"  Variant A (window=10, unweighted): AUC = {auc_10:.4f}")

    # Variant B: window=5, weighted
    print("\n  Rebuilding rolling features with window=5, weighted...")
    train_w5 = rebuild_rolling(train_df, window=5, weighted=True)
    test_w5 = rebuild_rolling(test_df, window=5, weighted=True)
    train_w5 = add_dc_features(train_w5, dc)
    test_w5 = add_dc_features(test_w5, dc)
    avail_w5 = [c for c in available if c in train_w5.columns]
    auc_w5 = train_and_eval(train_w5[avail_w5], y_train, test_w5[avail_w5], y_test)
    print(f"  Variant B (window=5, weighted):    AUC = {auc_w5:.4f}")

    print(f"\n  Summary:")
    print(f"    {'Variant':<35} {'AUC':>7} {'vs base':>8}")
    print(f"    {'-'*52}")
    print(f"    {'Current (5, unweighted)':<35} {auc_base:>7.4f} {'':>8}")
    print(f"    {'A: 10 games, unweighted':<35} {auc_10:>7.4f} {auc_10 - auc_base:>+8.4f}")
    print(f"    {'B: 5 games, linear decay':<35} {auc_w5:>7.4f} {auc_w5 - auc_base:>+8.4f}")


# =====================================================================
# 3. Feature ablation
# =====================================================================

def analysis_ablation(train_df, test_df):
    print("\n" + "=" * 60)
    print("  3. FEATURE GROUP ABLATION")
    print("=" * 60)

    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    # Group definitions
    dc_only = [c for c in DC_FEATURES if c in train_df.columns]

    formation_feats = [
        "h_formation_defenders", "h_formation_midfielders", "h_formation_forwards",
        "a_formation_defenders", "a_formation_midfielders", "a_formation_forwards",
        "fwd_vs_def", "def_vs_fwd",
        "formation_matchup_avg_goals",
    ]
    formation_feats = [c for c in formation_feats if c in train_df.columns]

    rolling_feats = [c for c in CONTEXT_FEATURES if "roll_" in c or "venue_" in c]
    rolling_feats = [c for c in rolling_feats if c in train_df.columns]

    h2h_tier_feats = [
        "h2h_avg_goals", "h_tier_score", "a_tier_score",
    ]
    h2h_tier_feats = [c for c in h2h_tier_feats if c in train_df.columns]

    experiments = [
        ("DC only", dc_only),
        ("DC + formation", dc_only + formation_feats),
        ("DC + rolling", dc_only + rolling_feats),
        ("DC + H2H + tier", dc_only + h2h_tier_feats),
        ("DC + all context (full model)", dc_only + [c for c in CONTEXT_FEATURES if c in train_df.columns]),
    ]

    print(f"\n  {'Model':<35} {'N feat':>7} {'AUC':>7}")
    print(f"  {'-'*52}")

    # DC standalone (no XGBoost)
    dc_proba = test_df["dc_prob_over25"].values
    dc_auc = roc_auc_score(y_test, dc_proba)
    print(f"  {'DC standalone (no XGBoost)':<35} {'--':>7} {dc_auc:>7.4f}")

    results = {}
    for name, feats in experiments:
        auc = train_and_eval(train_df[feats], y_train, test_df[feats], y_test)
        print(f"  {name:<35} {len(feats):>7} {auc:>7.4f}")
        results[name] = auc

    # Show what each group adds over DC alone
    print(f"\n  Marginal contribution over DC-only (AUC={results['DC only']:.4f}):")
    for name, auc in results.items():
        if name == "DC only":
            continue
        delta = auc - results["DC only"]
        print(f"    {name:<35} {delta:>+7.4f}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("Loading data and fitting Dixon-Coles...\n")
    train_df, test_df, dc = setup()

    analysis_correlation(train_df)
    analysis_ablation(train_df, test_df)
    analysis_rolling_windows(train_df, test_df)
