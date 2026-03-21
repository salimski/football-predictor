"""
Two-layer model for over/under 2.5 goals prediction.

Layer 1: Dixon-Coles (Poisson-based) — fitted from scratch via MLE.
         Outputs lambda_home, lambda_away, P(over/under 2.5), P(over/under 3.5).

Layer 2: XGBoost — learns to correct Dixon-Coles using the full pre-match
         feature set + DC outputs.  Outputs calibrated P(over 2.5).

Temporal split: train on earlier seasons, test on latest.
"""

import sqlite3
import os
import sys
import json

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss, brier_score_loss,
)
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.dixon_coles import DixonColesModel

# ---------------------------------------------------------------------------
# Feature columns — pre-match context (existing)
# ---------------------------------------------------------------------------

# Shared context features (non-odds, non-DC)
_SHARED_CONTEXT = [
    # Formation (parsed numbers)
    "h_formation_defenders", "h_formation_midfielders", "h_formation_forwards",
    "a_formation_defenders", "a_formation_midfielders", "a_formation_forwards",
    "fwd_vs_def", "def_vs_fwd",
    # Pre-match xG forecasts (from venue-specific rolling xG)
    "h_xg_forecast", "a_xg_forecast", "total_xg_forecast",
    # Pinnacle (2.5 line — correlated with all goal lines)
    "pinnacle_prob_over25", "pinnacle_dc_diff",
    # Understat formation xG (per-formation effectiveness)
    "h_formation_xg_per_game", "a_formation_xg_per_game",
    "h_formation_xga_per_game", "a_formation_xga_per_game",
    "formation_xg_matchup",
    # Understat top-player xG/xA concentration
    "h_top3_xg_per90", "a_top3_xg_per90",
    "h_top3_xa_per90", "a_top3_xa_per90",
]

# O/U 2.5 context features (original 2.5-line odds)
CONTEXT_FEATURES = _SHARED_CONTEXT + [
    "b365_prob_over25", "bb_avg_prob_over25", "odds_dc_diff",
]

# U3.5 context features (real 3.5-line odds — NaN in training, live at prediction time)
CONTEXT_FEATURES_U35 = _SHARED_CONTEXT + [
    "b365_prob_over35", "odds_dc_diff_35",
]

# U4.5 context features (real 4.5-line odds — NaN in training, live at prediction time)
CONTEXT_FEATURES_U45 = _SHARED_CONTEXT + [
    "b365_prob_over45", "odds_dc_diff_45",
]

# Dixon-Coles output features (added by Layer 1)
DC_FEATURES = [
    "dc_lambda_home", "dc_lambda_away",
    "dc_prob_over25", "dc_prob_under25",
    "dc_prob_over35", "dc_prob_under35",
    "dc_prob_over45", "dc_prob_under45",
]

TARGET_COL = "target_over25"
TARGET_U35 = "target_under35"
TARGET_U45 = "target_under45"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()
    return df


def temporal_split(df, test_season="2025/26"):
    train = df[df["season"] != test_season].copy()
    test = df[df["season"] == test_season].copy()
    return train, test


def add_dc_features(df, dc_model):
    """Generate Dixon-Coles predictions and attach as new columns."""
    preds = dc_model.predict_batch(
        df["home_team"].tolist(),
        df["away_team"].tolist(),
    )
    df["dc_lambda_home"]  = [p["lambda_home"]  for p in preds]
    df["dc_lambda_away"]  = [p["lambda_away"]  for p in preds]
    df["dc_prob_over25"]  = [p["prob_over25"]  for p in preds]
    df["dc_prob_under25"] = [p["prob_under25"] for p in preds]
    df["dc_prob_over35"]  = [p["prob_over35"]  for p in preds]
    df["dc_prob_under35"] = [p["prob_under35"] for p in preds]
    df["dc_prob_over45"]  = [p["prob_over45"]  for p in preds]
    df["dc_prob_under45"] = [p["prob_under45"] for p in preds]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_model(db_path, test_season="2025/26", model_dir="model/saved"):
    print("Loading features...")
    df = load_features(db_path)

    # Merge Understat formation + player xG features
    try:
        from features.player_features import compute_understat_features
        udf = compute_understat_features(db_path)
        if not udf.empty:
            df = df.merge(udf, on="match_id", how="left")
            print(f"Merged {len(udf.columns) - 1} Understat features")
    except Exception as exc:
        print(f"WARNING: Understat features unavailable: {exc}")

    print(f"Total rows: {len(df)}, columns: {len(df.columns)}")

    # Split
    train_df, test_df = temporal_split(df, test_season)
    print(f"Train: {len(train_df)} rows  ({sorted(train_df['season'].unique())})")
    print(f"Test:  {len(test_df)} rows  ({sorted(test_df['season'].unique())})")

    # ==================================================================
    # LAYER 1: Dixon-Coles
    # ==================================================================
    print("\n" + "=" * 60)
    print("  LAYER 1: Dixon-Coles (Poisson MLE)")
    print("=" * 60)

    dc = DixonColesModel(xi=0.0065)
    dc.fit(
        home_teams=train_df["home_team"].tolist(),
        away_teams=train_df["away_team"].tolist(),
        goals_home=train_df["goals_home"].tolist(),
        goals_away=train_df["goals_away"].tolist(),
        match_dates=train_df["match_date"].tolist(),
    )
    dc.print_params(top_n=8)

    # Dixon-Coles standalone accuracy on test set
    print("\n--- Dixon-Coles standalone (test set) ---")
    test_dc_preds = dc.predict_batch(
        test_df["home_team"].tolist(),
        test_df["away_team"].tolist(),
    )
    dc_proba_test = np.array([p["prob_over25"] for p in test_dc_preds])
    dc_pred_test = (dc_proba_test > 0.5).astype(int)
    y_test = test_df[TARGET_COL].values
    _print_metrics("DC", y_test, dc_pred_test, dc_proba_test)

    # Add DC features to both sets
    train_df = add_dc_features(train_df, dc)
    test_df = add_dc_features(test_df, dc)

    # Compute odds_dc_diff (market vs model disagreement)
    train_df["odds_dc_diff"] = train_df["b365_prob_over25"] - train_df["dc_prob_over25"]
    test_df["odds_dc_diff"] = test_df["b365_prob_over25"] - test_df["dc_prob_over25"]
    train_df["pinnacle_dc_diff"] = train_df["pinnacle_prob_over25"] - train_df["dc_prob_over25"]
    test_df["pinnacle_dc_diff"] = test_df["pinnacle_prob_over25"] - test_df["dc_prob_over25"]

    # Real 3.5/4.5 odds columns — NaN for all historical data (no source),
    # XGBoost handles NaN natively; real values provided at prediction time
    for frame in [train_df, test_df]:
        frame["b365_prob_over35"] = np.nan
        frame["odds_dc_diff_35"] = np.nan
        frame["b365_prob_over45"] = np.nan
        frame["odds_dc_diff_45"] = np.nan

    # ==================================================================
    # LAYER 2: XGBoost (O/U 2.5)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  LAYER 2: XGBoost — O/U 2.5 (corrects Dixon-Coles)")
    print("=" * 60)

    all_features = CONTEXT_FEATURES + DC_FEATURES
    available = [c for c in all_features if c in train_df.columns]
    missing = [c for c in all_features if c not in train_df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")

    X_train = train_df[available]
    y_train = train_df[TARGET_COL]
    X_test = test_df[available]

    print(f"\nFeatures used: {len(available)} ({len(CONTEXT_FEATURES)} context + {len(DC_FEATURES)} DC)")
    print(f"Train target: {y_train.mean():.3f} over rate")
    print(f"Test  target: {y_test.mean():.3f} over rate")

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    print("\n--- XGBoost O/U 2.5 (Layer 2, full pipeline) ---")
    _print_metrics("XGB-O25", y_test, y_pred_xgb, y_proba_xgb)

    # Probability calibration quality
    print(f"\n--- Probability Calibration (O/U 2.5) ---")
    print(f"  Brier score (DC alone):  {brier_score_loss(y_test, dc_proba_test):.4f}")
    print(f"  Brier score (XGB):       {brier_score_loss(y_test, y_proba_xgb):.4f}")
    print(f"  Log loss (DC alone):     {log_loss(y_test, dc_proba_test):.4f}")
    print(f"  Log loss (XGB):          {log_loss(y_test, y_proba_xgb):.4f}")

    # Feature importance
    print("\n--- Feature Importance O/U 2.5 (top 15) ---")
    importances = pd.Series(
        xgb_model.feature_importances_, index=available
    ).sort_values(ascending=False)
    for feat, imp in importances.head(15).items():
        print(f"  {imp:.4f}  {feat}")

    # ==================================================================
    # LAYER 2b: XGBoost — Under 3.5 (line-specific odds)
    # ==================================================================
    features_u35 = CONTEXT_FEATURES_U35 + DC_FEATURES
    avail_u35 = [c for c in features_u35 if c in train_df.columns]
    missing_u35 = [c for c in features_u35 if c not in train_df.columns]
    if missing_u35:
        print(f"WARNING: Missing U3.5 features: {missing_u35}")

    xgb_u35, imp_u35 = _train_target(
        "Under 3.5", TARGET_U35, train_df, test_df, avail_u35,
        dc_proba_col="dc_prob_under35",
    )

    # ==================================================================
    # LAYER 2c: XGBoost — Under 4.5 (line-specific odds)
    # ==================================================================
    features_u45 = CONTEXT_FEATURES_U45 + DC_FEATURES
    avail_u45 = [c for c in features_u45 if c in train_df.columns]
    missing_u45 = [c for c in features_u45 if c not in train_df.columns]
    if missing_u45:
        print(f"WARNING: Missing U4.5 features: {missing_u45}")

    xgb_u45, imp_u45 = _train_target(
        "Under 4.5", TARGET_U45, train_df, test_df, avail_u45,
        dc_proba_col="dc_prob_under45",
    )

    # ==================================================================
    # Save
    # ==================================================================
    os.makedirs(model_dir, exist_ok=True)
    dc.save(os.path.join(model_dir, "dixon_coles.json"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgb_model.joblib"))
    joblib.dump(xgb_u35, os.path.join(model_dir, "xgb_under35.joblib"))
    joblib.dump(xgb_u45, os.path.join(model_dir, "xgb_under45.joblib"))

    meta = {
        "features": available,
        "features_u35": avail_u35,
        "features_u45": avail_u45,
        "context_features": CONTEXT_FEATURES,
        "dc_features": DC_FEATURES,
        "train_medians": X_train.median().to_dict(),
        "test_season": test_season,
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nModels saved to {model_dir}/")
    return dc, xgb_model, importances


def _train_target(label, target_col, train_df, test_df, features,
                  dc_proba_col):
    """Train an XGBoost model for a given target and report metrics."""
    print("\n" + "=" * 60)
    print(f"  LAYER 2: XGBoost — {label}")
    print("=" * 60)

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col].values

    base_rate = y_train.mean()
    test_rate = y_test.mean()
    print(f"\nTrain base rate ({label}): {base_rate:.3f}")
    print(f"Test  base rate ({label}): {test_rate:.3f}")
    print(f"Naive baseline accuracy (always predict majority): {max(test_rate, 1-test_rate):.3f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- XGBoost {label} ---")
    _print_metrics(f"XGB-{label}", y_test, y_pred, y_proba)

    # DC standalone comparison
    dc_proba = test_df[dc_proba_col].values
    dc_pred = (dc_proba > 0.5).astype(int)
    print(f"\n--- DC standalone {label} ---")
    _print_metrics(f"DC-{label}", y_test, dc_pred, dc_proba)

    # Calibration comparison
    print(f"\n--- Calibration Comparison {label} ---")
    print(f"  Brier (DC):  {brier_score_loss(y_test, dc_proba):.4f}")
    print(f"  Brier (XGB): {brier_score_loss(y_test, y_proba):.4f}")
    print(f"  LogLoss (DC):  {log_loss(y_test, dc_proba):.4f}")
    print(f"  LogLoss (XGB): {log_loss(y_test, y_proba):.4f}")

    # Feature importance
    print(f"\n--- Feature Importance {label} (top 10) ---")
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, val in imp.head(10).items():
        print(f"  {val:.4f}  {feat}")

    return model, imp


def _print_metrics(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  ROC AUC:   {auc:.3f}")
    print(f"  Confusion matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
