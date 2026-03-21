"""
Phase 2 entry point: build and write the features table.

Usage:
    python build_features.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import DB_PATH
from features.builder import build, write_features

df = build(DB_PATH)
write_features(df, DB_PATH)

print()
print("=== Summary ===")
print(f"Rows         : {len(df)}")
print(f"Columns      : {len(df.columns)}")

total = len(df)
over  = df["target_over25"].sum()
under = total - over
print(f"\ntarget_over25 distribution:")
print(f"  Over  2.5 : {over:>5}  ({100*over/total:.1f}%)")
print(f"  Under 2.5 : {under:>5}  ({100*under/total:.1f}%)")

u35 = df["target_under35"].sum()
print(f"\ntarget_under35 distribution:")
print(f"  Under 3.5 : {u35:>5}  ({100*u35/total:.1f}%)")
print(f"  Over  3.5 : {total-u35:>5}  ({100*(total-u35)/total:.1f}%)")

u45 = df["target_under45"].sum()
print(f"\ntarget_under45 distribution:")
print(f"  Under 4.5 : {u45:>5}  ({100*u45/total:.1f}%)")
print(f"  Over  4.5 : {total-u45:>5}  ({100*(total-u45)/total:.1f}%)")

print(f"\nh_roll_games_available distribution:")
print(df["h_roll_games_available"].value_counts().sort_index().to_string())

null_xg = df["pm_home_xg"].isna().sum()
print(f"\npm_home_xg NULL count : {null_xg}  (expected 8)")

null_target = df["target_over25"].isna().sum()
print(f"target_over25 NULLs  : {null_target}  (expected 0)")

# New features check
for col in ["h_tier_score", "formation_matchup_avg_goals", "h2h_avg_goals",
            "h_venue_goals_scored", "fwd_vs_def", "h_formation_forwards",
            "h_xg_forecast", "a_xg_forecast", "total_xg_forecast",
            "h_venue_xg_for", "a_venue_xg_for"]:
    null_ct = df[col].isna().sum()
    print(f"{col} NULLs: {null_ct}")

print()
print("=== Verification Checks ===")

# Check 1: row count
assert len(df) == 1456, f"FAIL row count: {len(df)}"
print(f"[OK] Row count = 1456")

# Check 2: target distribution ~60.9%
rate = over / total
assert 0.58 < rate < 0.64, f"FAIL over rate: {rate:.3f}"
print(f"[OK] Over 2.5 rate = {rate:.3f} (expected ~0.609)")

# Check 3: round 1 of 2021/22 has games_available = 0
r1 = df[(df["round_number"] == 1) & (df["season"] == "2021/22")]
assert (r1["h_roll_games_available"] == 0).all(), "FAIL: R1 2021/22 has non-zero games_available"
print(f"[OK] Round 1 2021/22 h_roll_games_available = 0 for all {len(r1)} rows")

# Check 4: round 1 of 2022/23 has games_available 1-5 (carry-over)
r1_next = df[(df["round_number"] == 1) & (df["season"] == "2022/23")]
assert (r1_next["h_roll_games_available"].between(1, 5)).all(), \
    f"FAIL: R1 2022/23 games_available out of range"
print(f"[OK] Round 1 2022/23 h_roll_games_available in [1,5] for all {len(r1_next)} rows")

# Check 5: exactly 8 null pm_home_xg
assert null_xg == 8, f"FAIL pm_home_xg NULLs: {null_xg}"
print(f"[OK] pm_home_xg NULL count = 8")

# Check 6: over25_rate and wins in [0, 1]
for col in ["h_roll_over25_rate", "h_roll_wins"]:
    s = df[col].dropna()
    assert (s >= 0).all() and (s <= 1).all(), f"FAIL {col} out of [0,1]"
print(f"[OK] h_roll_over25_rate and h_roll_wins in [0.0, 1.0]")

# Check 7: no 2021/22 rows have standing features
assert (df[df["season"] == "2021/22"]["h_standing_rank"].isna()).all(), \
    "FAIL: 2021/22 has non-null h_standing_rank"
print(f"[OK] 2021/22 rows have NULL h_standing_rank")

# Check 8: tier scores are always present (no NULLs)
assert df["h_tier_score"].notna().all(), "FAIL: h_tier_score has NULLs"
assert df["a_tier_score"].notna().all(), "FAIL: a_tier_score has NULLs"
print(f"[OK] Tier scores have no NULLs")

# Check 9: fwd_vs_def exists
assert "fwd_vs_def" in df.columns, "FAIL: fwd_vs_def missing"
assert "def_vs_fwd" in df.columns, "FAIL: def_vs_fwd missing"
print(f"[OK] Formation extras (fwd_vs_def, def_vs_fwd) present")

print()
print("All checks passed.")
