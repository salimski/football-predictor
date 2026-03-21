"""
Temporal validation + calibration analysis.

1. Train on 2021/22-2023/24, test on 2024/25 + 2025/26.
2. Calibration plot (reliability diagram) with ECE.
3. If miscalibrated, apply isotonic/Platt recalibration and replot.
"""

import sqlite3
import os
import sys
import json

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, log_loss, brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DB_PATH
from model.dixon_coles import DixonColesModel
from model.train import (
    CONTEXT_FEATURES, DC_FEATURES, TARGET_COL,
    load_features, add_dc_features,
)


def print_metrics(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    ll = log_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print(f"  [{name}]")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  ROC AUC:   {auc:.3f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  Log loss:  {ll:.4f}")
    print(f"  Confusion matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    return {"acc": acc, "auc": auc, "brier": brier, "logloss": ll}


def compute_ece(y_true, y_proba, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    bin_data = []
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            bin_data.append((bin_edges[i], bin_edges[i + 1], 0, 0, 0))
            continue
        avg_pred = y_proba[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_true - avg_pred)
        bin_data.append((bin_edges[i], bin_edges[i + 1], n_bin, avg_pred, avg_true))
    return ece, bin_data


def plot_calibration(y_true, y_proba, label, filename, title_suffix=""):
    """Reliability diagram + histogram of predictions."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    ece, bin_data = compute_ece(y_true, y_proba, n_bins=10)

    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax1.plot(prob_pred, prob_true, "s-", color="#2196F3", linewidth=2, markersize=8, label=label)

    # Annotate bins with sample counts
    for pp, pt in zip(prob_pred, prob_true):
        ax1.annotate(f"{pt:.0%}", (pp, pt), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#333")

    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives (actual)")
    ax1.set_title(f"Calibration Plot{title_suffix}\nECE = {ece:.4f}")
    ax1.legend(loc="lower right")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    ax2.hist(y_proba, bins=20, range=(0, 1), color="#2196F3", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of predicted probabilities")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")
    return ece


def main():
    print("=" * 60)
    print("  TEMPORAL VALIDATION")
    print("=" * 60)

    # Load data
    df = load_features(DB_PATH)
    print(f"Total: {len(df)} rows")

    train_seasons = ["2021/22", "2022/23", "2023/24"]
    test_seasons = ["2024/25", "2025/26"]

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    print(f"Train: {len(train_df)} rows  {train_seasons}")
    print(f"Test:  {len(test_df)} rows   {test_seasons}")
    print(f"Train over rate: {train_df[TARGET_COL].mean():.3f}")
    print(f"Test  over rate: {test_df[TARGET_COL].mean():.3f}")

    # ── Layer 1: Dixon-Coles ──────────────────────────────────────────────
    print("\n--- Fitting Dixon-Coles on training seasons only ---")
    dc = DixonColesModel(xi=0.0065)
    dc.fit(
        home_teams=train_df["home_team"].tolist(),
        away_teams=train_df["away_team"].tolist(),
        goals_home=train_df["goals_home"].tolist(),
        goals_away=train_df["goals_away"].tolist(),
        match_dates=train_df["match_date"].tolist(),
    )
    dc.print_params(top_n=5)

    # Add DC features
    train_df = add_dc_features(train_df, dc)
    test_df = add_dc_features(test_df, dc)

    # Compute odds_dc_diff (market vs model disagreement)
    train_df["odds_dc_diff"] = train_df["b365_prob_over25"] - train_df["dc_prob_over25"]
    test_df["odds_dc_diff"] = test_df["b365_prob_over25"] - test_df["dc_prob_over25"]
    train_df["pinnacle_dc_diff"] = train_df["pinnacle_prob_over25"] - train_df["dc_prob_over25"]
    test_df["pinnacle_dc_diff"] = test_df["pinnacle_prob_over25"] - test_df["dc_prob_over25"]

    # DC standalone on test
    print("\n--- Dixon-Coles standalone (test: 24/25 + 25/26) ---")
    y_test = test_df[TARGET_COL].values
    dc_proba = test_df["dc_prob_over25"].values
    dc_pred = (dc_proba > 0.5).astype(int)
    dc_metrics = print_metrics("DC", y_test, dc_pred, dc_proba)

    # ── Layer 2: XGBoost ──────────────────────────────────────────────────
    all_features = CONTEXT_FEATURES + DC_FEATURES
    available = [c for c in all_features if c in train_df.columns]

    X_train = train_df[available]
    y_train = train_df[TARGET_COL]
    X_test = test_df[available]

    print(f"\n--- Training XGBoost ({len(available)} features) ---")
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

    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_proba_xgb > 0.5).astype(int)

    print("\n--- XGBoost full pipeline (test: 24/25 + 25/26) ---")
    xgb_metrics = print_metrics("XGB", y_test, y_pred_xgb, y_proba_xgb)

    # Per-season breakdown
    for season in test_seasons:
        mask = test_df["season"].values == season
        if mask.sum() == 0:
            continue
        yt = y_test[mask]
        yp = y_proba_xgb[mask]
        ypred = (yp > 0.5).astype(int)
        print(f"\n  --- {season} only ({mask.sum()} matches) ---")
        print_metrics(f"XGB-{season}", yt, ypred, yp)

    # ══════════════════════════════════════════════════════════════════════
    # CALIBRATION ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  CALIBRATION ANALYSIS")
    print("=" * 60)

    os.makedirs("model/plots", exist_ok=True)

    # Raw calibration
    ece_raw = plot_calibration(
        y_test, y_proba_xgb, "XGBoost (raw)",
        "model/plots/calibration_raw.png",
        " - Raw XGBoost"
    )
    print(f"\n  ECE (raw XGBoost): {ece_raw:.4f}")

    # Detailed bin report
    ece, bin_data = compute_ece(y_test, y_proba_xgb, n_bins=10)
    print(f"\n  {'Bin':>12}  {'N':>5}  {'Avg Pred':>9}  {'Avg True':>9}  {'Gap':>7}")
    print(f"  {'-'*48}")
    for lo, hi, n, avg_p, avg_t in bin_data:
        gap = abs(avg_t - avg_p) if n > 0 else 0
        label = f"{lo:.0%}-{hi:.0%}"
        if n > 0:
            print(f"  {label:>12}  {n:>5}  {avg_p:>9.3f}  {avg_t:>9.3f}  {gap:>7.3f}")
        else:
            print(f"  {label:>12}  {n:>5}       -         -        -")

    # ── Isotonic recalibration ────────────────────────────────────────────
    print("\n--- Applying isotonic regression recalibration ---")

    # Fit calibrator on training predictions
    train_proba = xgb_model.predict_proba(X_train)[:, 1]

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train_proba, y_train)

    y_proba_iso = iso.predict(y_proba_xgb)
    y_pred_iso = (y_proba_iso > 0.5).astype(int)

    print("\n--- XGBoost + Isotonic (test: 24/25 + 25/26) ---")
    iso_metrics = print_metrics("XGB+Iso", y_test, y_pred_iso, y_proba_iso)

    ece_iso = plot_calibration(
        y_test, y_proba_iso, "XGBoost + Isotonic",
        "model/plots/calibration_isotonic.png",
        " - After Isotonic Recalibration"
    )
    print(f"\n  ECE (after isotonic): {ece_iso:.4f}")

    # ── Platt scaling ─────────────────────────────────────────────────────
    print("\n--- Applying Platt scaling recalibration ---")

    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(max_iter=1000)
    platt.fit(train_proba.reshape(-1, 1), y_train)

    y_proba_platt = platt.predict_proba(y_proba_xgb.reshape(-1, 1))[:, 1]
    y_pred_platt = (y_proba_platt > 0.5).astype(int)

    print("\n--- XGBoost + Platt (test: 24/25 + 25/26) ---")
    platt_metrics = print_metrics("XGB+Platt", y_test, y_pred_platt, y_proba_platt)

    ece_platt = plot_calibration(
        y_test, y_proba_platt, "XGBoost + Platt",
        "model/plots/calibration_platt.png",
        " - After Platt Scaling"
    )
    print(f"\n  ECE (after Platt): {ece_platt:.4f}")

    # ── Combined plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    for proba, label, color in [
        (y_proba_xgb, f"Raw (ECE={ece_raw:.4f})", "#F44336"),
        (y_proba_iso, f"Isotonic (ECE={ece_iso:.4f})", "#2196F3"),
        (y_proba_platt, f"Platt (ECE={ece_platt:.4f})", "#4CAF50"),
    ]:
        pt, pp = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        ax.plot(pp, pt, "s-", color=color, linewidth=2, markersize=7, label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Calibration Comparison (test: 2024/25 + 2025/26)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model/plots/calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: model/plots/calibration_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(f"\n  {'Model':<20} {'Acc':>6} {'AUC':>6} {'Brier':>7} {'LogLoss':>8} {'ECE':>7}")
    print(f"  {'-'*55}")
    print(f"  {'DC alone':<20} {dc_metrics['acc']:>6.3f} {dc_metrics['auc']:>6.3f} {dc_metrics['brier']:>7.4f} {dc_metrics['logloss']:>8.4f}      -")
    print(f"  {'XGB (raw)':<20} {xgb_metrics['acc']:>6.3f} {xgb_metrics['auc']:>6.3f} {xgb_metrics['brier']:>7.4f} {xgb_metrics['logloss']:>8.4f} {ece_raw:>7.4f}")
    print(f"  {'XGB + Isotonic':<20} {iso_metrics['acc']:>6.3f} {iso_metrics['auc']:>6.3f} {iso_metrics['brier']:>7.4f} {iso_metrics['logloss']:>8.4f} {ece_iso:>7.4f}")
    print(f"  {'XGB + Platt':<20} {platt_metrics['acc']:>6.3f} {platt_metrics['auc']:>6.3f} {platt_metrics['brier']:>7.4f} {platt_metrics['logloss']:>8.4f} {ece_platt:>7.4f}")

    # Recommend best calibration
    best = min(
        [("raw", ece_raw, xgb_metrics),
         ("isotonic", ece_iso, iso_metrics),
         ("platt", ece_platt, platt_metrics)],
        key=lambda x: x[1]
    )
    print(f"\n  Best calibration: {best[0]} (ECE={best[1]:.4f})")

    # Save best calibrator if it's not raw
    if best[0] == "isotonic":
        joblib.dump(iso, "model/saved/calibrator.joblib")
        meta_path = "model/saved/meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["calibrator"] = "isotonic"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print("  Saved isotonic calibrator to model/saved/calibrator.joblib")
    elif best[0] == "platt":
        joblib.dump(platt, "model/saved/calibrator.joblib")
        meta_path = "model/saved/meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["calibrator"] = "platt"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print("  Saved Platt calibrator to model/saved/calibrator.joblib")
    else:
        print("  Raw XGBoost is best calibrated -- no post-hoc calibrator needed.")


if __name__ == "__main__":
    main()
