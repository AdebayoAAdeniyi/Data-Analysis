"""
SQL analysis of the European Soccer Database (SQLite) with Python.

Data: ``database.sqlite`` from the Kaggle data set "European Soccer Database"
(tables Country, League, Team, Match, Player, Player_Attributes,
Team_Attributes; seasons 2008/2009 to 2015/2016). Download it from Kaggle and
pass its path with ``--db``.

What this script shows, from basic to advanced SQL
--------------------------------------------------
1. Inspecting a schema (``sqlite_master``)
2. SELECT, JOIN, ORDER BY, LIMIT
3. Self joins with aliases (the Team table is joined twice: home and away)
4. GROUP BY / HAVING for league statistics per season
5. Subqueries that avoid a many to many "fan out" (players x attribute rows)
6. CASE expressions to bin values
7. Window functions (RANK, AVG OVER) to compare leagues and seasons
8. Common table expressions (WITH) for a home advantage analysis

Usage::

    python european_soccer_sql.py --db data/database.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

BIG_FIVE = "('Spain', 'Germany', 'France', 'Italy', 'England')"

QUERIES: dict[str, str] = {
    "tables": """
        SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name""",

    "countries": "SELECT * FROM Country",

    "leagues": """
        SELECT l.id, l.name AS league, c.name AS country
        FROM League AS l
        JOIN Country AS c ON c.id = l.country_id""",

    "teams_first_20": """
        SELECT team_api_id, team_long_name, team_short_name
        FROM Team
        ORDER BY team_long_name
        LIMIT 20""",

    # Team is joined twice, once for the home side and once for the away side
    "first_spanish_matches": """
        SELECT m.id, c.name AS country, l.name AS league, m.season, m.stage, m.date,
               ht.team_long_name AS home_team, at.team_long_name AS away_team,
               m.home_team_goal, m.away_team_goal
        FROM Match AS m
        JOIN Country AS c ON c.id = m.country_id
        JOIN League  AS l ON l.id = m.league_id
        LEFT JOIN Team AS ht ON ht.team_api_id = m.home_team_api_id
        LEFT JOIN Team AS at ON at.team_api_id = m.away_team_api_id
        WHERE c.name = 'Spain'
        ORDER BY m.date
        LIMIT 10""",

    "league_season_stats": f"""
        SELECT c.name AS country, l.name AS league, m.season,
               COUNT(DISTINCT m.stage)                       AS stages,
               COUNT(DISTINCT m.home_team_api_id)            AS teams,
               ROUND(AVG(m.home_team_goal), 3)               AS avg_home_goals,
               ROUND(AVG(m.away_team_goal), 3)               AS avg_away_goals,
               ROUND(AVG(m.home_team_goal - m.away_team_goal), 3) AS avg_goal_diff,
               ROUND(AVG(m.home_team_goal + m.away_team_goal), 3) AS avg_goals,
               SUM(m.home_team_goal + m.away_team_goal)      AS total_goals
        FROM Match AS m
        JOIN Country AS c ON c.id = m.country_id
        JOIN League  AS l ON l.id = m.league_id
        WHERE c.name IN {BIG_FIVE}
        GROUP BY c.name, l.name, m.season
        HAVING COUNT(DISTINCT m.stage) > 10
        ORDER BY c.name, m.season DESC""",

    # Window functions: rank the leagues inside each season by goals per game
    "league_rank_by_goals": f"""
        WITH per_season AS (
            SELECT l.name AS league, m.season,
                   AVG(m.home_team_goal + m.away_team_goal) AS avg_goals
            FROM Match AS m
            JOIN League  AS l ON l.id = m.league_id
            JOIN Country AS c ON c.id = m.country_id
            WHERE c.name IN {BIG_FIVE}
            GROUP BY l.name, m.season
        )
        SELECT season, league, ROUND(avg_goals, 3) AS avg_goals,
               RANK() OVER (PARTITION BY season ORDER BY avg_goals DESC) AS rank_in_season,
               ROUND(avg_goals - AVG(avg_goals) OVER (PARTITION BY season), 3)
                   AS diff_from_season_mean
        FROM per_season
        ORDER BY season, rank_in_season""",

    # CTE + CASE: share of home wins, draws and away wins per league
    "home_advantage": f"""
        WITH results AS (
            SELECT l.name AS league,
                   CASE WHEN m.home_team_goal > m.away_team_goal THEN 'home_win'
                        WHEN m.home_team_goal = m.away_team_goal THEN 'draw'
                        ELSE 'away_win' END AS result
            FROM Match AS m
            JOIN League  AS l ON l.id = m.league_id
            JOIN Country AS c ON c.id = m.country_id
            WHERE c.name IN {BIG_FIVE}
        )
        SELECT league, COUNT(*) AS matches,
               ROUND(100.0 * SUM(result = 'home_win') / COUNT(*), 1) AS home_win_pct,
               ROUND(100.0 * SUM(result = 'draw')     / COUNT(*), 1) AS draw_pct,
               ROUND(100.0 * SUM(result = 'away_win') / COUNT(*), 1) AS away_win_pct
        FROM results
        GROUP BY league
        ORDER BY home_win_pct DESC""",

    # Player has one row per player, Player_Attributes many rows per player.
    # Joining them directly would repeat each player once per attribute row
    # (a fan out), so the attributes are averaged per player in a subquery first.
    # Height is stored in cm and weight in pounds; weight is converted to kg.
    "height_vs_rating": """
        SELECT CASE
                   WHEN ROUND(p.height) < 165 THEN 165
                   WHEN ROUND(p.height) > 195 THEN 195
                   ELSE ROUND(p.height)
               END                                     AS height_cm,
               COUNT(*)                                AS players,
               ROUND(AVG(pa.avg_overall_rating), 2)    AS avg_overall_rating,
               ROUND(AVG(pa.avg_potential), 2)         AS avg_potential,
               ROUND(AVG(p.weight) * 0.45359237, 1)    AS avg_weight_kg
        FROM Player AS p
        LEFT JOIN (
            SELECT player_api_id,
                   AVG(overall_rating) AS avg_overall_rating,
                   AVG(potential)      AS avg_potential
            FROM Player_Attributes
            GROUP BY player_api_id
        ) AS pa ON pa.player_api_id = p.player_api_id
        GROUP BY height_cm
        ORDER BY height_cm""",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=Path("data/database.sqlite"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")

    results = {}
    with sqlite3.connect(args.db) as conn:
        for name, sql in QUERIES.items():
            df = pd.read_sql_query(sql, conn)
            results[name] = df
            print(f"\n=== {name} ({len(df)} rows) ===")
            print(df.head(40).to_string(index=False))

    args.results.mkdir(parents=True, exist_ok=True)
    for name in ["league_season_stats", "league_rank_by_goals", "home_advantage",
                 "height_vs_rating"]:
        results[name].to_csv(args.results / f"soccer_{name}.csv", index=False)

    args.figures.mkdir(parents=True, exist_ok=True)
    stats = results["league_season_stats"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for league, g in stats.groupby("league"):
        g = g.sort_values("season")
        ax.plot(g["season"], g["avg_goals"], marker="o", label=league)
    ax.set_ylabel("Goals per match")
    ax.set_title("Average goals per match, big five leagues")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.savefig(args.figures / "soccer_goals_per_match.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    h = results["height_vs_rating"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(h["height_cm"], h["avg_overall_rating"], marker="o", label="Overall rating")
    ax.plot(h["height_cm"], h["avg_potential"], marker="s", label="Potential")
    ax.set_xlabel("Height (cm, ends clipped at 165 and 195)")
    ax.set_ylabel("Average FIFA rating")
    ax.set_title("Player rating by height")
    ax.legend()
    fig.savefig(args.figures / "soccer_rating_by_height.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTables saved to {args.results}/, figures saved to {args.figures}/")


if __name__ == "__main__":
    main()
