-- Analytical queries on the `matches` table (SQLite). Loaded by src/analysis.py.
-- Blocks are separated by "-- name: <id>".

-- name: home_advantage
-- Share of home wins / draws / away wins and goals per match, by season.
SELECT season,
       COUNT(*) AS matches,
       ROUND(100.0 * SUM(result = 'H') / COUNT(*), 1) AS home_win_pct,
       ROUND(100.0 * SUM(result = 'D') / COUNT(*), 1) AS draw_pct,
       ROUND(100.0 * SUM(result = 'A') / COUNT(*), 1) AS away_win_pct,
       ROUND(1.0 * SUM(hg + ag) / COUNT(*), 2) AS goals_per_match
FROM matches
GROUP BY season
ORDER BY season;

-- name: final_table
-- League table per season using a UNION of home and away rows, then a window rank.
WITH rows AS (
  SELECT season, home AS team, hg AS gf, ag AS ga,
         CASE WHEN hg > ag THEN 3 WHEN hg = ag THEN 1 ELSE 0 END AS pts FROM matches
  UNION ALL
  SELECT season, away, ag, hg,
         CASE WHEN ag > hg THEN 3 WHEN ag = hg THEN 1 ELSE 0 END FROM matches
), t AS (
  SELECT season, team, COUNT(*) AS played, SUM(pts) AS points, SUM(gf) AS gf, SUM(ga) AS ga,
         SUM(gf) - SUM(ga) AS gd
  FROM rows GROUP BY season, team
)
SELECT season, RANK() OVER (PARTITION BY season ORDER BY points DESC, gd DESC, gf DESC) AS pos,
       team, played, points, gf, ga, gd
FROM t
ORDER BY season, pos;

-- name: rolling_form
-- Points in the last 5 matches before each fixture, per team (window frame).
WITH rows AS (
  SELECT match_id, date, home AS team,
         CASE WHEN hg > ag THEN 3 WHEN hg = ag THEN 1 ELSE 0 END AS pts FROM matches
  UNION ALL
  SELECT match_id, date, away,
         CASE WHEN ag > hg THEN 3 WHEN ag = hg THEN 1 ELSE 0 END FROM matches
)
SELECT match_id, date, team, pts,
       SUM(pts) OVER (PARTITION BY team ORDER BY date, match_id
                      ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS form_last5
FROM rows
ORDER BY team, date;

-- name: biggest_overperformers
-- Points actually won minus points expected from bookmaker average odds (implied, margin removed).
WITH p AS (
  SELECT season, home AS team,
         3.0 * (1/odds_h) / (1/odds_h + 1/odds_d + 1/odds_a) + 1.0 * (1/odds_d) / (1/odds_h + 1/odds_d + 1/odds_a) AS xpts,
         CASE WHEN hg > ag THEN 3 WHEN hg = ag THEN 1 ELSE 0 END AS pts
  FROM matches WHERE odds_h IS NOT NULL
  UNION ALL
  SELECT season, away,
         3.0 * (1/odds_a) / (1/odds_h + 1/odds_d + 1/odds_a) + 1.0 * (1/odds_d) / (1/odds_h + 1/odds_d + 1/odds_a),
         CASE WHEN ag > hg THEN 3 WHEN ag = hg THEN 1 ELSE 0 END
  FROM matches WHERE odds_h IS NOT NULL
)
SELECT season, team, ROUND(SUM(pts), 0) AS points, ROUND(SUM(xpts), 1) AS market_expected_points,
       ROUND(SUM(pts) - SUM(xpts), 1) AS diff
FROM p GROUP BY season, team
ORDER BY diff DESC
LIMIT 10;
