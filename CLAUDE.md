# Bundesliga Over/Under 2.5 Goals Predictor

## Purpose

Binary classifier predicting whether a Bundesliga match will have over or under 2.5 total goals. Uses a two-layer architecture (Dixon-Coles Poisson model + XGBoost) trained on 5 seasons of data (2021/22-2025/26). The output is compared against Polymarket implied probabilities to find betting edges, with an 8% edge threshold for BET signals.

## Folder Structure

```
football-predictor/
├── config.py                   # API keys, LEAGUE_ID=78 (Bundesliga), SEASONS, DB_PATH
├── requirements.txt            # requests, aiofiles, pandas, numpy
├── predict.py                  # Interactive CLI predictor (DC + XGB + blending + starting XI)
├── dashboard.py                # Streamlit UI: match predictor + tracker (4 metrics, Polymarket edge)
├── build_features.py           # Entry point: builds features table from raw DB
├── train_model.py              # Entry point: calls model/train.py
├── run_collection.py           # Entry point: runs all collectors
├── audit_db.py                 # DB verification utility
├── eda.ipynb                   # Exploratory data analysis notebook
│
├── collector/
│   ├── apifootball.py          # api-sports.io client (fixtures, stats, lineups, standings)
│   ├── understat.py            # Understat.com scraper (match-level xG data)
│   ├── understat_advanced.py   # Understat.com scraper (player xG/xA + formation stats)
│   ├── player_stats.py         # api-sports.io player stats collector (squad ratings)
│   ├── odds_collector.py       # football-data.co.uk CSV parser (B365, Pinnacle, BbAvg odds)
│   └── normalize.py            # Team name canonicalization across all data sources
│
├── features/
│   ├── builder.py              # All feature engineering: rolling stats (5-game window, strict
│   │                             temporal ordering), venue-specific xG rolling, formation parsing,
│   │                             standings from prior season, xG forecasts, odds merge
│   └── player_features.py      # Squad rating features, Understat formation/player xG features,
│                                 starting XI fuzzy matching, typical XI loader
│
├── model/
│   ├── dixon_coles.py          # Layer 1: bivariate Poisson MLE with rho correction, time decay
│   ├── train.py                # Layer 2: XGBoost training, temporal split, feature importance
│   ├── validate.py             # Strict validation (train 21-24, test 24-26), calibration plots
│   ├── diagnose.py             # Model diagnostic utilities
│   ├── saved/
│   │   ├── dixon_coles.json    # Fitted DC params (attack/defense per team, home_adv, rho)
│   │   ├── xgb_model.joblib    # Trained XGBoost classifier (O/U 2.5)
│   │   ├── xgb_under35.joblib  # Trained XGBoost classifier (U3.5)
│   │   ├── xgb_under45.joblib  # Trained XGBoost classifier (U4.5)
│   │   └── meta.json           # Feature list (33), train medians, test_season
│   └── plots/
│       └── calibration_*.png   # Reliability diagrams (raw, isotonic, platt, comparison)
│
├── data/
│   ├── raw.db                  # SQLite: matches, statistics, lineups, standings, xg, odds,
│   │                             features, player_stats, understat_player_stats,
│   │                             understat_formation_stats
│   ├── prediction_tracker.json # Live prediction log with results and ROI tracking
│   └── external/
│       └── football_data_co_uk/  # CSVs per season: 21_22(1).csv through 25_26(1).csv
```

## Database Tables (data/raw.db)

| Table | Rows | Source | Key columns |
|---|---|---|---|
| matches | ~1456 | api-sports.io | match_id, season, home_team, away_team, raw_json |
| statistics | ~1456 | api-sports.io | match_id, raw_json (16 stats per side) |
| lineups | ~1456 | api-sports.io | match_id, raw_json (formations, players) |
| standings | 5 | api-sports.io | season, raw_json (final league table) |
| xg | ~1448 | Understat | match_date, home/away_team, home/away_xg, raw_json |
| odds | ~1458 | football-data.co.uk | match_date, home/away_team, b365/pinnacle/bbavg probs |
| features | ~1456 | builder.py | All engineered features, targets, identities |
| player_stats | ~523 | api-sports.io | player_id, team_id, season, rating, appearances, minutes |
| understat_player_stats | ~2152 | Understat | player_id, team_name, season, xg, xa, xg_per90, xa_per90 |
| understat_formation_stats | ~807 | Understat | team_name, season, formation, xg_for/against per game |

## Model Architecture

### Layer 1: Dixon-Coles (1997)

Bivariate Poisson with team-specific attack/defense parameters:
- `log(lambda_home) = home_adv + attack[home] + defense[away]`
- `log(lambda_away) = attack[away] + defense[home]`
- Low-score correction factor rho for (0,0), (0,1), (1,0), (1,1) scorelines
- Exponential time decay: xi=0.0065 per day (~halving weight every ~106 days)
- Outputs: lambda_home, lambda_away, full scoreline probability matrix, P(over/under 2.5), P(over/under 3.5), P(over/under 4.5)

### Layer 2: Three XGBoost Models

All three share the same hyperparameters but have **per-model feature lists** stored in `meta.json`:

```python
XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss"
)
```

| Model | File | Target | Base Rate | Features |
|---|---|---|---|---|
| O/U 2.5 | `xgb_model.joblib` | `target_over25` (total_goals > 2) | 60.9% | 33 (2.5-line odds) |
| Under 3.5 | `xgb_under35.joblib` | `target_under35` (total_goals <= 3) | 59.6% | 32 (3.5-line odds, NaN in training) |
| Under 4.5 | `xgb_under45.joblib` | `target_under45` (total_goals <= 4) | 78.7% | 32 (4.5-line odds, NaN in training) |

### Per-Model Feature Lists

Each model gets its own feature columns (stored in `meta.json` as `features`, `features_u35`, `features_u45`). The shared base is 22 features (formation, xG forecast, Pinnacle, formation xG, top player xG). The difference is in odds features:

| Model | Odds Features | Rationale |
|---|---|---|
| O/U 2.5 | b365_prob_over25, bb_avg_prob_over25, odds_dc_diff | Direct 2.5-line odds (from football-data.co.uk historically, user input at prediction time) |
| Under 3.5 | b365_prob_over35, odds_dc_diff_35 | Real 3.5-line odds (NaN in training, user provides B365 under 3.5 at prediction time) |
| Under 4.5 | b365_prob_over45, odds_dc_diff_45 | Real 4.5-line odds (NaN in training, user provides B365 under 4.5 at prediction time) |

### Odds at Prediction Time

The dashboard and CLI accept B365 odds for all three lines:
- **Over 2.5 odds** (decimal, e.g. 1.85) → `b365_prob_over25 = 1 / odds`
- **Under 3.5 odds** (decimal, e.g. 1.45) → `b365_prob_over35 = 1 - (1 / odds)` (derive over from under)
- **Under 4.5 odds** (decimal, e.g. 1.20) → `b365_prob_over45 = 1 - (1 / odds)` (derive over from under)

All odds features are NaN during training (football-data.co.uk only has 2.5-line odds for Bundesliga). XGBoost handles NaN natively — the model learns without odds during training and uses them when provided at prediction time. This is the same pattern as xG forecast features.

The Lines Panel shows an odds source indicator per line: "live odds" when user provided real B365 odds, "DC only" when no odds available.

### Feature Set Details

| Group | Count | Features |
|---|---|---|
| Formation | 8 | h/a_formation_defenders/midfielders/forwards, fwd_vs_def, def_vs_fwd |
| xG Forecast | 3 | h/a_xg_forecast, total_xg_forecast (from venue-specific rolling) |
| Odds (2.5) | 3 | b365_prob_over25, bb_avg_prob_over25, odds_dc_diff (O/U 2.5 model only) |
| Odds (3.5) | 2 | b365_prob_over35, odds_dc_diff_35 (U3.5 model only) |
| Odds (4.5) | 2 | b365_prob_over45, odds_dc_diff_45 (U4.5 model only) |
| Pinnacle | 2 | pinnacle_prob_over25, pinnacle_dc_diff |
| Formation xG | 5 | h/a_formation_xg_per_game, h/a_formation_xga_per_game, formation_xg_matchup |
| Top Player xG | 4 | h/a_top3_xg_per90, h/a_top3_xa_per90 |
| DC Outputs | 8 | dc_lambda_home/away, dc_prob_over25/under25, dc_prob_over35/under35, dc_prob_over45/under45 |

### Train/Test Split

- **Production model** (train.py): Train on 2021/22-2024/25 (1231 rows), test on 2025/26 (225 rows)
- **Strict validation** (validate.py): Train on 2021/22-2023/24, test on 2024/25 + 2025/26

### Performance

| Model | Production AUC | Strict AUC | DC Standalone AUC |
|---|---|---|---|
| O/U 2.5 | 0.608 | 0.598 | 0.577 |
| Under 3.5 | 0.616 | 0.601 | 0.605 (prod) / 0.593 (strict) |
| Under 4.5 | 0.569 | 0.602 | 0.625 (prod) / 0.601 (strict) |

- O/U 2.5 AUC evolution: 0.543 -> 0.574 -> 0.585 -> 0.596 -> 0.608 (33 features)
- U3.5 and U4.5 odds features are NaN during training (no historical 3.5/4.5 odds available). At prediction time, real B365 under 3.5 / under 4.5 odds are provided — XGBoost handles the NaN-to-real transition natively.
- DC has better calibration than XGBoost on all targets, which the blending formula leverages.
- Target distributions: O2.5 60.9%, U3.5 59.6%, U4.5 78.7%

### Data Sources for Understat Features

- **Formation xG** (`understat_formation_stats`): Scraped from `getTeamData/{slug}/{year}` API. Contains per-formation xG for/against totals for every team-season. ~807 rows across 5 seasons.
- **Player xG/xA** (`understat_player_stats`): Scraped from `getLeagueData/Bundesliga/{year}` API. Contains per-player season xG, xA, shots, key passes. ~2152 rows (players with ≥90 min). Top 3 players by xG/90 per team used as features.
- **Collection script:** `collector/understat_advanced.py` — run to refresh both tables.

## Blending Formula

All three lines (O/U 2.5, U3.5, U4.5) use the same dynamic blending formula. XGBoost probability is blended with DC probability using a weighted average that penalizes large model divergence:

```python
gap = abs(model_prob - dc_prob)
dc_weight = 0.5 + (gap * 0.5)    # range: [0.50, 1.00]
model_weight = 1 - dc_weight      # range: [0.00, 0.50]
blended = (dc_prob * dc_weight) + (model_prob * model_weight)
```

Applied identically to all three lines:
- `blended_over25 = blend(xgb_over25, dc_prob_over25)`
- `blended_under35 = blend(xgb_under35, dc_prob_under35)`
- `blended_under45 = blend(xgb_under45, dc_prob_under45)`

**Why it was added:** DC has better calibration than XGBoost on all targets. The formula anchors predictions toward DC when models disagree. At 0pp gap: 50/50 blend. At 25pp gap: DC gets 62.5%. At 50pp gap: DC gets 75%.

## Prediction Pipeline (predict.py / dashboard.py)

1. Load DC + 3x XGBoost models + per-model feature lists from `model/saved/`
2. Load Understat lookups (formation xG, player xG) from DB
3. Accept starting XI declarations (optional) for both teams
4. Accept B365 odds for all three lines (optional): over 2.5, under 3.5, under 4.5
5. Build feature row (superset): formation parsing, venue-specific xG rolling, odds conversion (under→over for 3.5/4.5), Understat formation/player xG (XI-specific when lineup declared), DC predictions
6. Each XGBoost model selects its own feature columns from the superset
7. Dynamic blending with DC probability
8. Edge = blended_prob - polymarket_implied
9. Signal logic (O/U 2.5):
   - **BET OVER**: edge > 8% AND blended_prob > 58%
   - **BET UNDER**: edge < -8% AND blended_prob < 58%
   - **PASS (model favors over)**: edge < -8% BUT blended_prob >= 58% — "Model still favors over — edge insufficient to bet against"
   - **PASS**: edge within ±8%
10. Overconfidence warning if blended_prob > 80%

### Under Betting Threshold Rule

BET UNDER 2.5 requires blended_prob < 58%. Rationale: if the blended model still assigns >58% chance of going over, the negative edge against Polymarket is unreliable — the model genuinely thinks the match is more likely to go over, so betting under is fighting the model's own conviction.

### Three-Line Panel

All three lines use dedicated XGBoost models blended with Dixon-Coles:

| Line | DC Prob | Model/Blended | Polymarket | Signal |
|---|---|---|---|---|
| O/U 2.5 | DC P(O2.5) | blended_over25 | user input | BET/PASS |
| U 3.5 | DC P(U3.5) | blended_under35 | user input | BET/PASS |
| U 4.5 | DC P(U4.5) | blended_under45 | user input | BET/PASS |

### Betting Thresholds

| Line | Blended Threshold | Edge Threshold | Rationale |
|---|---|---|---|
| BET OVER 2.5 | blended > 58% | edge > 8% | Model must favor over |
| BET UNDER 2.5 | blended < 58% | edge < -8% | Model must favor under |
| BET U3.5 | blended > 62% | edge > 8% | Higher base rate (59.6%) requires higher conviction |
| BET U4.5 | blended > 78% | edge > 8% | Very high base rate (78.7%) requires near-certainty |

- Tracker logs all three lines independently with separate bet_placed/result fields per line.

## Starting XI Lineup System

### Input
Both `predict.py` (CLI) and `dashboard.py` (Streamlit) accept starting XI declarations for each team:
- 11 player names per team (e.g., "Kane", "Musiala", "Kimmich")
- "Load typical XI" button auto-fills with the 11 most frequently started players from the `lineups` table for that team-season
- XI input is optional — when omitted, team season averages are used (original behavior)

### Fuzzy Matching (features/player_features.py)
Player names are matched against `understat_player_stats.player_name` using a multi-strategy approach:
1. **Unicode normalization** — strips accents, converts ß->ss, handles U+FFFD replacement chars. Enables "Gross" to match "Pascal Groß", "Sane" to match "Leroy Sané"
2. **Exact last-name match** — single-word input matched against last word of DB name. Prevents "Kim" from matching "Kimmich" (gets "Kim Min-Jae" instead)
3. **Substring match** — "Kane" matches "Harry Kane" (requires 4+ chars and unique match)
4. **difflib fuzzy match** — fallback with cutoff=0.6 for typos and abbreviations

### Feature Computation Priority (per player)
1. **XI-specific lookup** from `understat_player_stats` in current season (most accurate)
2. **Prior season lookup** from `understat_player_stats` if player not in current season (e.g., mid-season transfer)
3. **Team season average** fallback if fewer than 6 of 11 players matched (insufficient XI coverage)

### How XI Affects Features
When XI is declared, `h_top3_xg_per90` / `a_top3_xg_per90` / `h_top3_xa_per90` / `a_top3_xa_per90` are computed from the actual declared players' stats rather than the team's top 3 across the whole squad. This means:
- A weaker lineup (rotation) produces lower xG features → model adjusts probability down
- A stronger lineup produces higher features → model adjusts up
- Formation xG features are unaffected (still based on declared formation)

## Dashboard (dashboard.py)

- **Screen 1 (Match Predictor):** Team/formation dropdowns, starting XI inputs (11 per team in 3-column grid), "Load typical XI" button, B365 odds, Polymarket price. **Formation Intelligence** section showing per-formation xG/game and combined matchup total. Shows 4 metrics side by side: DC, Model, Blended (with DC weight caption), B365 implied. XI matching details in expandable section. Scoreline heatmap. Polymarket edge analysis with BET/PASS signal.
- **Screen 2 (Model Tracker):** Editable prediction log (date, match, our_prob, dc_prob, blended_prob, b365, poly_price, edge, signal, bet_placed, result). Running stats: hit rate, ROI, avg edge.
- **Run:** `streamlit run dashboard.py --server.headless true`
- Session state pattern: prediction results stored in `st.session_state["last_prediction"]` to survive Streamlit reruns.

## Known Issues and Weaknesses

1. **Pinnacle NULL rate:** 39% of 2025/26 matches have NULL Pinnacle odds (P>2.5 column missing from football-data.co.uk CSVs for recent matches). XGBoost handles NaN but this hurts the feature's usefulness.
2. **Small sample for live tracking:** Only 9 predictions logged so far (all from one matchday, 2026-03-17). All 9 were under — insufficient to validate calibration or ROI.
3. **U3.5/U4.5 calibration:** XGBoost U3.5 and U4.5 models have higher Brier scores than DC standalone. The blending formula mitigates this by anchoring toward DC.
4. **Rolling features not in XGBoost:** The 50+ rolling features (h_roll_goals_scored, etc.) exist in the features table but were stripped during feature selection down to the 31-feature set. They may still add value.
5. **Calibration overconfidence:** Model tends to be overconfident in the 70-85% range. The blending formula partially mitigates this but doesn't fully solve it.
6. **Tracker records before blending:** The 9 existing tracker entries don't have a `blended_prob` field (they predate the blending deployment).
7. **No automated data refresh:** Collection, feature building, and retraining are manual processes.
8. **Understat formation data uses cumulative season stats:** Formation xG features are season-to-date totals, not match-by-match rolling. Early-season predictions use prior season fallback.

## What Was Most Recently Changed (2026-03-20)

1. **Three-model architecture** — Dedicated XGBoost models for U3.5 and U4.5 targets, with per-model feature lists. All three use DC+XGBoost blending.
2. **Real odds for all three lines** — Dashboard and CLI accept B365 under 3.5 and under 4.5 odds. Converted via `P(over) = 1 - 1/under_odds`. Odds features are NaN during training (no historical source), real at prediction time. Synthetic proxy removed.
3. **Per-model feature lists** — `meta.json` stores `features` (33), `features_u35` (32), `features_u45` (32). Both predict.py and dashboard.py select correct columns per model.
4. **33-feature set** — Added `dc_prob_over45` and `dc_prob_under45` to DC features (up from 31 to 33).
5. **Betting thresholds per line** — O/U 2.5: blended ≷ 58%, U3.5: blended > 62%, U4.5: blended > 78%. All require 8% edge minimum.
6. **Under betting threshold** — BET UNDER 2.5 requires blended_prob < 58%. Prevents betting under when model favors over.
7. **Three-line panel** — Dashboard shows all three lines with blended probabilities, Polymarket edge, odds source indicator, and signals.
8. **Tracker expanded** — Logs U3.5 and U4.5 bets independently with blended_prob, edge, signal, bet_placed, result per line.

### Prior Changes (2026-03-19)

1. **Starting XI lineup system** — both CLI and dashboard accept full starting XI declarations (11 players per team). Player names are fuzzy-matched against Understat data using unicode normalization + multi-strategy matching. Top-3 xG/xA features are recomputed from the actual declared lineup rather than team season averages.
2. **"Load typical XI" button** — auto-populates with the 11 most frequently started players from the lineups table for the selected team-season.
3. **Injury adjustment removed** — both hardcoded and data-driven versions removed from predict.py and dashboard.py. Starting XI declaration makes it redundant.
4. **31-feature model deployed** — AUC improved from 0.580 to 0.596 (production) / 0.598 (strict)
5. **Understat formation xG features** (5 new) — per-formation xG for/against per game from `understat_formation_stats` table.
6. **Understat top-player xG/xA features** (4 new) — average xG/90 and xA/90 of top 3 players per team from `understat_player_stats` table.

## Logical Next Steps

1. **Hyperparameter tuning for U3.5/U4.5:** The U3.5 and U4.5 models use the same hyperparams as O/U 2.5. Targeted tuning (especially min_child_weight for the imbalanced U4.5 target) could improve AUC.
2. **Live tracking and validation:** Continue logging predictions across multiple matchdays to validate calibration and ROI with statistical significance (need 50+ bets minimum).
3. **Retrain after full season:** Once 2025/26 completes (June 2026), retrain on all 5 seasons. This gives the model 300+ more training rows and fills in Pinnacle NULL gaps for completed matches.
4. **Rolling features re-evaluation:** Test adding top rolling features (h_roll_goals_scored, h_roll_over25_rate, etc.) back into the model — they were cut early but may improve AUC with the current 31-feature baseline.
5. **Automated pipeline:** Schedule collection + feature build + retrain as a cron job or GitHub Action so the model stays current without manual intervention.
6. **Calibration post-processing:** If live tracking confirms overconfidence, apply isotonic recalibration (validate.py already has the infrastructure).
7. **Match-level formation xG:** Currently formation xG uses cumulative season data. Could compute rolling formation xG (last N matches in that formation) for better temporal sensitivity.
