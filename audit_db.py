import sqlite3
import json
import sys

DB_PATH = r"C:\Users\Salim\predict\football-predictor\data\raw.db"

def sep(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row
cur = con.cursor()

# ─────────────────────────────────────────────────────────────
# 1. COMPLETENESS — row counts per season in matches
# ─────────────────────────────────────────────────────────────
sep("CHECK 1 — COMPLETENESS: row counts per season in matches")
rows = cur.execute("""
    SELECT season, COUNT(*) as matches
    FROM matches
    GROUP BY season
    ORDER BY season
""").fetchall()
print(f"{'Season':<12} {'Matches':>8}")
print("-"*22)
for r in rows:
    print(f"{r['season']:<12} {r['matches']:>8}")
total = sum(r['matches'] for r in rows)
print(f"{'TOTAL':<12} {total:>8}")

# Also show counts for other tables
for tbl in ('statistics', 'lineups', 'standings', 'xg'):
    try:
        n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl}: {n} rows total")
    except Exception as e:
        print(f"  {tbl}: ERROR — {e}")

# ─────────────────────────────────────────────────────────────
# 2. SCORE SANITY
# ─────────────────────────────────────────────────────────────
sep("CHECK 2 — SCORE SANITY")

null_home = cur.execute("""
    SELECT COUNT(*) FROM matches
    WHERE json_extract(raw_json,'$.goals.home') IS NULL
""").fetchone()[0]
null_away = cur.execute("""
    SELECT COUNT(*) FROM matches
    WHERE json_extract(raw_json,'$.goals.away') IS NULL
""").fetchone()[0]
print(f"NULL goals.home: {null_home}")
print(f"NULL goals.away: {null_away}")

neg = cur.execute("""
    SELECT match_id, season,
           json_extract(raw_json,'$.teams.home.name') as home,
           json_extract(raw_json,'$.teams.away.name') as away,
           json_extract(raw_json,'$.goals.home') as gh,
           json_extract(raw_json,'$.goals.away') as ga,
           json_extract(raw_json,'$.fixture.date') as dt
    FROM matches
    WHERE CAST(json_extract(raw_json,'$.goals.home') AS INTEGER) < 0
       OR CAST(json_extract(raw_json,'$.goals.away') AS INTEGER) < 0
""").fetchall()
print(f"\nNegative goals rows: {len(neg)}")
for r in neg:
    print(f"  {r['dt'][:10]} {r['home']} {r['gh']}-{r['ga']} {r['away']}  [{r['season']}]")

high = cur.execute("""
    SELECT match_id, season,
           json_extract(raw_json,'$.teams.home.name') as home,
           json_extract(raw_json,'$.teams.away.name') as away,
           json_extract(raw_json,'$.goals.home') as gh,
           json_extract(raw_json,'$.goals.away') as ga,
           json_extract(raw_json,'$.fixture.date') as dt
    FROM matches
    WHERE CAST(json_extract(raw_json,'$.goals.home') AS INTEGER) > 10
       OR CAST(json_extract(raw_json,'$.goals.away') AS INTEGER) > 10
""").fetchall()
print(f"\nGoals > 10 rows: {len(high)}")
for r in high:
    print(f"  {r['dt'][:10]} {r['home']} {r['gh']}-{r['ga']} {r['away']}  [{r['season']}]")

# Show rows where goals are non-NULL but look suspicious (e.g. both 0 across many games just as a sanity peek)
sample = cur.execute("""
    SELECT json_extract(raw_json,'$.goals.home') as gh,
           json_extract(raw_json,'$.goals.away') as ga,
           COUNT(*) as cnt
    FROM matches
    WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL
    GROUP BY gh, ga
    ORDER BY cnt DESC
    LIMIT 10
""").fetchall()
print("\nTop 10 scorelines (home-away : count):")
for r in sample:
    print(f"  {r['gh']}-{r['ga']} : {r['cnt']}")

# ─────────────────────────────────────────────────────────────
# 3. DATE RANGE
# ─────────────────────────────────────────────────────────────
sep("CHECK 3 — DATE RANGE")
r = cur.execute("""
    SELECT MIN(dt) as earliest, MAX(dt) as latest, COUNT(*) as total
    FROM (
        SELECT json_extract(raw_json,'$.fixture.date') as dt
        FROM matches
        WHERE json_extract(raw_json,'$.fixture.date') IS NOT NULL
    )
""").fetchone()
print(f"Earliest match: {r['earliest']}")
print(f"Latest match:   {r['latest']}")
print(f"Rows with date: {r['total']}")

# ─────────────────────────────────────────────────────────────
# 4. STATISTICS COVERAGE
# ─────────────────────────────────────────────────────────────
sep("CHECK 4 — STATISTICS COVERAGE")

total_matches = cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
total_stats   = cur.execute("SELECT COUNT(*) FROM statistics").fetchone()[0]
print(f"Total matches rows: {total_matches}")
print(f"Total statistics rows: {total_stats}")

checks = [
    ("shots_home",       "statistics", "json_extract(raw_json,'$[0].statistics[0].value')"),
    ("shots_away",       "statistics", "json_extract(raw_json,'$[1].statistics[0].value')"),
    ("possession_home",  "statistics", "json_extract(raw_json,'$[0].statistics[9].value')"),
    ("possession_away",  "statistics", "json_extract(raw_json,'$[1].statistics[9].value')"),
    ("halftime_home",    "matches",    "json_extract(raw_json,'$.score.halftime.home')"),
    ("halftime_away",    "matches",    "json_extract(raw_json,'$.score.halftime.away')"),
]

print(f"\n{'Field':<20} {'NULLs':>8} {'Total':>8} {'% NULL':>8}  Status")
print("-"*56)
for name, tbl, expr in checks:
    n_null = cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {expr} IS NULL").fetchone()[0]
    n_total = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    pct = 100.0 * n_null / n_total if n_total else 0
    flag = "WARN >" if pct > 5 else "OK"
    print(f"{name:<20} {n_null:>8} {n_total:>8} {pct:>7.1f}%  {flag}")

# ─────────────────────────────────────────────────────────────
# 5. FORMATION COVERAGE
# ─────────────────────────────────────────────────────────────
sep("CHECK 5 — FORMATION COVERAGE")

total_lineups = cur.execute("SELECT COUNT(*) FROM lineups").fetchone()[0]
print(f"Total lineups rows: {total_lineups}")

distinct_formations = cur.execute("""
    SELECT COUNT(DISTINCT formation) FROM (
        SELECT json_extract(raw_json,'$[0].formation') as formation FROM lineups
        UNION ALL
        SELECT json_extract(raw_json,'$[1].formation') as formation FROM lineups
    )
""").fetchone()[0]
print(f"Distinct formations (home+away combined): {distinct_formations}")

top10 = cur.execute("""
    SELECT formation, COUNT(*) as cnt FROM (
        SELECT json_extract(raw_json,'$[0].formation') as formation FROM lineups
        UNION ALL
        SELECT json_extract(raw_json,'$[1].formation') as formation FROM lineups
    )
    WHERE formation IS NOT NULL
    GROUP BY formation
    ORDER BY cnt DESC
    LIMIT 10
""").fetchall()
print("\nTop 10 formations:")
for r in top10:
    print(f"  {r['formation']:<12} {r['cnt']:>5}")

null_formation = cur.execute("""
    SELECT COUNT(*) FROM lineups
    WHERE json_extract(raw_json,'$[0].formation') IS NULL
       OR json_extract(raw_json,'$[1].formation') IS NULL
""").fetchone()[0]
print(f"\nMatches with NULL home OR away formation: {null_formation}")
pct_null = 100.0 * null_formation / total_lineups if total_lineups else 0
print(f"  ({pct_null:.1f}% of lineups rows)")

# ─────────────────────────────────────────────────────────────
# 6. XG JOIN TEST
# ─────────────────────────────────────────────────────────────
sep("CHECK 6 — XG JOIN TEST")

total_m = cur.execute("SELECT COUNT(*) FROM matches WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL").fetchone()[0]
total_xg = cur.execute("SELECT COUNT(*) FROM xg").fetchone()[0]
print(f"Matches with non-NULL goals: {total_m}")
print(f"XG rows: {total_xg}")

joined = cur.execute("""
    SELECT COUNT(*) FROM matches m
    INNER JOIN xg x
      ON substr(json_extract(m.raw_json,'$.fixture.date'),1,10) = x.match_date
     AND json_extract(m.raw_json,'$.teams.home.name') = x.home_team
     AND json_extract(m.raw_json,'$.teams.away.name') = x.away_team
""").fetchone()[0]

total_all = cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
failed = total_all - joined
pct_fail = 100.0 * failed / total_all if total_all else 0
print(f"\nTotal matches (all):   {total_all}")
print(f"Successful joins:      {joined}")
print(f"Failed joins:          {failed}  ({pct_fail:.1f}%)")

# Show sample of unmatched (up to 10) — what dates/names don't join
unmatched = cur.execute("""
    SELECT substr(json_extract(m.raw_json,'$.fixture.date'),1,10) as dt,
           json_extract(m.raw_json,'$.teams.home.name') as home,
           json_extract(m.raw_json,'$.teams.away.name') as away,
           m.season
    FROM matches m
    WHERE NOT EXISTS (
        SELECT 1 FROM xg x
        WHERE substr(json_extract(m.raw_json,'$.fixture.date'),1,10) = x.match_date
          AND json_extract(m.raw_json,'$.teams.home.name') = x.home_team
          AND json_extract(m.raw_json,'$.teams.away.name') = x.away_team
    )
    LIMIT 10
""").fetchall()
if unmatched:
    print(f"\nSample unmatched matches (up to 10):")
    for r in unmatched:
        print(f"  {r['dt']}  {r['home']} vs {r['away']}  [{r['season']}]")

# ─────────────────────────────────────────────────────────────
# 7. TEAM NAME CONSISTENCY
# ─────────────────────────────────────────────────────────────
sep("CHECK 7 — TEAM NAME CONSISTENCY (unique home teams)")

teams = cur.execute("""
    SELECT DISTINCT json_extract(raw_json,'$.teams.home.name') as team
    FROM matches
    ORDER BY team
""").fetchall()
print(f"Distinct home team names: {len(teams)}")
print()
for i, r in enumerate(teams, 1):
    print(f"  {i:>2}. {r['team']}")

if len(teams) != 18:
    print(f"\n  WARNING: Expected 18 teams, found {len(teams)}")
else:
    print(f"\n  OK: Exactly 18 distinct home teams.")

# ─────────────────────────────────────────────────────────────
# 8. GOAL DISTRIBUTION
# ─────────────────────────────────────────────────────────────
sep("CHECK 8 — GOAL DISTRIBUTION")

dist = cur.execute("""
    SELECT
        CASE
            WHEN total >= 8 THEN '8+'
            ELSE CAST(total AS TEXT)
        END as bucket,
        COUNT(*) as cnt
    FROM (
        SELECT
            CAST(json_extract(raw_json,'$.goals.home') AS INTEGER) +
            CAST(json_extract(raw_json,'$.goals.away') AS INTEGER) as total
        FROM matches
        WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL
          AND json_extract(raw_json,'$.goals.away') IS NOT NULL
    )
    GROUP BY bucket
    ORDER BY
        CASE bucket WHEN '8+' THEN 99 ELSE CAST(bucket AS INTEGER) END
""").fetchall()

total_with_goals = sum(r['cnt'] for r in dist)
print(f"Matches with goals data: {total_with_goals}")
print(f"\n{'Goals':>6} {'Count':>7} {'%':>7}")
print("-"*22)
for r in dist:
    pct = 100.0 * r['cnt'] / total_with_goals
    print(f"{r['bucket']:>6} {r['cnt']:>7} {pct:>6.1f}%")

over25 = cur.execute("""
    SELECT COUNT(*) FROM matches
    WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL
      AND json_extract(raw_json,'$.goals.away') IS NOT NULL
      AND (CAST(json_extract(raw_json,'$.goals.home') AS INTEGER) +
           CAST(json_extract(raw_json,'$.goals.away') AS INTEGER)) > 2.5
""").fetchone()[0]
under25 = total_with_goals - over25
print(f"\nOver 2.5 goals:  {over25}/{total_with_goals} = {100.0*over25/total_with_goals:.1f}%")
print(f"Under 2.5 goals: {under25}/{total_with_goals} = {100.0*under25/total_with_goals:.1f}%")

print("\nOver 2.5 rate by season:")
by_season = cur.execute("""
    SELECT season,
           COUNT(*) as total,
           SUM(CASE WHEN (CAST(json_extract(raw_json,'$.goals.home') AS INTEGER) +
                          CAST(json_extract(raw_json,'$.goals.away') AS INTEGER)) > 2.5
                    THEN 1 ELSE 0 END) as over25
    FROM matches
    WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL
      AND json_extract(raw_json,'$.goals.away') IS NOT NULL
    GROUP BY season
    ORDER BY season
""").fetchall()
print(f"  {'Season':<12} {'Total':>6} {'O2.5':>6} {'Rate':>7}")
print("  " + "-"*34)
for r in by_season:
    rate = 100.0 * r['over25'] / r['total'] if r['total'] else 0
    print(f"  {r['season']:<12} {r['total']:>6} {r['over25']:>6} {rate:>6.1f}%")

# ─────────────────────────────────────────────────────────────
# 9. HOME ADVANTAGE
# ─────────────────────────────────────────────────────────────
sep("CHECK 9 — HOME ADVANTAGE")

overall = cur.execute("""
    SELECT
        AVG(CAST(json_extract(raw_json,'$.goals.home') AS REAL)) as avg_home,
        AVG(CAST(json_extract(raw_json,'$.goals.away') AS REAL)) as avg_away
    FROM matches
    WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL
      AND json_extract(raw_json,'$.goals.away') IS NOT NULL
""").fetchone()
home_adv = overall['avg_home'] - overall['avg_away']
print(f"Overall avg goals scored:")
print(f"  Home: {overall['avg_home']:.3f}")
print(f"  Away: {overall['avg_away']:.3f}")
print(f"  Home advantage (diff): {home_adv:+.3f}")

by_season2 = cur.execute("""
    SELECT season,
           AVG(CAST(json_extract(raw_json,'$.goals.home') AS REAL)) as avg_home,
           AVG(CAST(json_extract(raw_json,'$.goals.away') AS REAL)) as avg_away
    FROM matches
    WHERE json_extract(raw_json,'$.goals.home') IS NOT NULL
      AND json_extract(raw_json,'$.goals.away') IS NOT NULL
    GROUP BY season
    ORDER BY season
""").fetchall()
print(f"\n  {'Season':<12} {'Avg Home':>9} {'Avg Away':>9} {'Advantage':>10}")
print("  " + "-"*43)
for r in by_season2:
    adv = r['avg_home'] - r['avg_away']
    print(f"  {r['season']:<12} {r['avg_home']:>9.3f} {r['avg_away']:>9.3f} {adv:>+10.3f}")

# ─────────────────────────────────────────────────────────────
# 10. STANDINGS PARSE TEST — season 2023/24
# ─────────────────────────────────────────────────────────────
sep("CHECK 10 — STANDINGS PARSE TEST (season 2023/24)")

seasons_available = cur.execute("SELECT DISTINCT season FROM standings ORDER BY season").fetchall()
print(f"Seasons in standings table: {[r['season'] for r in seasons_available]}")

target = "2023/24"
row = cur.execute("SELECT raw_json FROM standings WHERE season = ?", (target,)).fetchone()

if not row:
    # Try alternate format
    row = cur.execute("SELECT raw_json, season FROM standings ORDER BY season DESC LIMIT 1").fetchone()
    if row:
        target = row['season']
        print(f"Season '2023/24' not found, using latest: {target}")
    else:
        print("No standings data found!")
        sys.exit(0)

data = json.loads(row['raw_json'])

# Navigate to the standings list
# Structure: data is response list; response[0].league.standings[0] is the table
try:
    table = data[0]['league']['standings'][0]
except (KeyError, IndexError, TypeError):
    # Maybe it's already the list directly
    try:
        table = data[0]['standings'][0]
    except Exception:
        table = data

print(f"\nStandings table for season {target} — {len(table)} teams\n")
header = f"{'Rk':>3} {'Team':<28} {'Pts':>4} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GF':>4} {'GA':>4} {'GD':>4} {'Form':<6}"
print(header)
print("-" * len(header))

for t in table:
    rank      = t.get('rank', '?')
    name      = t.get('team', {}).get('name', '?')
    points    = t.get('points', '?')
    all_stats = t.get('all', {})
    played    = all_stats.get('played', '?')
    wins      = all_stats.get('win', '?')
    draws     = all_stats.get('draw', '?')
    losses    = all_stats.get('lose', '?')
    gf        = all_stats.get('goals', {}).get('for', '?')
    ga        = all_stats.get('goals', {}).get('against', '?')
    gd        = t.get('goalsDiff', '?')
    form      = t.get('form', '')
    print(f"{rank:>3} {name:<28} {points:>4} {played:>3} {wins:>3} {draws:>3} {losses:>3} {gf:>4} {ga:>4} {gd:>4} {form:<6}")

con.close()
print("\n" + "="*70)
print("  AUDIT COMPLETE")
print("="*70)
