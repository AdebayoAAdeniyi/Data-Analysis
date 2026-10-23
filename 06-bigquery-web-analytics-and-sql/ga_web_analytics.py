"""
Web analytics of the Google Merchandise Store with SQL on Google BigQuery.

Data: the public BigQuery data set ``bigquery-public-data.google_analytics_sample``
(one table per day, ``ga_sessions_20160801`` to ``ga_sessions_20170801``, one row
per website session in the Google Analytics 360 export format, with nested
RECORD columns such as ``totals``, ``device``, ``geoNetwork`` and
``trafficSource``).

Business questions answered
---------------------------
1. Where do the visitors come from, how many are new, and in which countries
   do they actually buy? (users, new users, buyers, conversion rate, revenue)
2. How is engagement (page views) distributed over continents and
   sub continents?
3. Which devices and marketing channels convert visitors into buyers?
4. How did traffic and sales develop month by month?

Good BigQuery practice shown here
---------------------------------
* Every query is first run as a *dry run* to estimate the bytes it will scan
  (BigQuery bills by bytes scanned), and real runs are capped with
  ``maximum_bytes_billed`` so a mistake can never produce a large bill.
* Wildcard tables (``ga_sessions_*``) with ``_TABLE_SUFFIX`` filters, so only
  the needed days are scanned.
* Nested fields are read with dot notation (``totals.transactions``).

Requirements::

    pip install google-cloud-bigquery db-dtypes pandas matplotlib plotly
    gcloud auth application-default login      # or GOOGLE_APPLICATION_CREDENTIALS

Usage::

    python ga_web_analytics.py --project YOUR_GCP_PROJECT            # run everything
    python ga_web_analytics.py --project YOUR_GCP_PROJECT --dry-run  # only cost estimates
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TABLE = "`bigquery-public-data.google_analytics_sample.ga_sessions_*`"
DATE_FILTER = "_TABLE_SUFFIX BETWEEN '20160801' AND '20170801'"

# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #
QUERIES: dict[str, str] = {
    # 1. Users, new users and buyers by country
    "users_by_country": f"""
SELECT
  geoNetwork.country AS country,
  COUNT(DISTINCT fullVisitorId) AS users,
  COUNT(DISTINCT IF(visitNumber = 1, fullVisitorId, NULL)) AS new_users,
  ROUND(100 * COUNT(DISTINCT IF(visitNumber = 1, fullVisitorId, NULL))
            / COUNT(DISTINCT fullVisitorId), 2) AS new_users_pct,
  COUNT(DISTINCT IF(totals.transactions > 0, fullVisitorId, NULL)) AS buyers,
  ROUND(100 * COUNT(DISTINCT IF(totals.transactions > 0, fullVisitorId, NULL))
            / COUNT(DISTINCT fullVisitorId), 3) AS conversion_rate_pct,
  SUM(IFNULL(totals.transactions, 0)) AS transactions,
  ROUND(SUM(IFNULL(totals.totalTransactionRevenue, 0)) / 1e6, 2) AS revenue_usd
FROM {TABLE}
WHERE {DATE_FILTER}
GROUP BY country
ORDER BY users DESC
LIMIT 15""",

    # 2a. Page views by continent
    "pageviews_by_continent": f"""
SELECT
  geoNetwork.continent AS continent,
  SUM(totals.pageviews) AS total_pageviews,
  ROUND(100 * SUM(totals.pageviews) / SUM(SUM(totals.pageviews)) OVER (), 2) AS share_pct
FROM {TABLE}
WHERE {DATE_FILTER}
GROUP BY continent
ORDER BY total_pageviews DESC""",

    # 2b. Page views by sub continent, with the Americas split properly into
    # North and South America using the sub continent field
    "pageviews_by_subcontinent": f"""
SELECT
  geoNetwork.subContinent AS sub_continent,
  CASE
    WHEN geoNetwork.subContinent IN ('Northern America', 'Central America', 'Caribbean')
      THEN 'North America'
    WHEN geoNetwork.subContinent = 'South America' THEN 'South America'
    ELSE geoNetwork.continent
  END AS region,
  SUM(totals.pageviews) AS total_pageviews
FROM {TABLE}
WHERE {DATE_FILTER}
GROUP BY sub_continent, region
ORDER BY total_pageviews DESC""",

    # 3a. Conversion by device category
    "conversion_by_device": f"""
SELECT
  device.deviceCategory AS device,
  COUNT(*) AS sessions,
  COUNTIF(totals.transactions > 0) AS sessions_with_purchase,
  ROUND(100 * COUNTIF(totals.transactions > 0) / COUNT(*), 3) AS session_conversion_pct,
  ROUND(AVG(IFNULL(totals.pageviews, 0)), 2) AS avg_pageviews
FROM {TABLE}
WHERE {DATE_FILTER}
GROUP BY device
ORDER BY sessions DESC""",

    # 3b. Conversion by marketing channel
    "conversion_by_channel": f"""
SELECT
  channelGrouping AS channel,
  COUNT(*) AS sessions,
  COUNTIF(totals.transactions > 0) AS sessions_with_purchase,
  ROUND(100 * COUNTIF(totals.transactions > 0) / COUNT(*), 3) AS session_conversion_pct,
  ROUND(SUM(IFNULL(totals.totalTransactionRevenue, 0)) / 1e6, 2) AS revenue_usd
FROM {TABLE}
WHERE {DATE_FILTER}
GROUP BY channel
ORDER BY revenue_usd DESC""",

    # 4. Monthly trend
    "monthly_trend": f"""
SELECT
  FORMAT_DATE('%Y-%m', PARSE_DATE('%Y%m%d', date)) AS month,
  COUNT(*) AS sessions,
  COUNT(DISTINCT fullVisitorId) AS users,
  SUM(IFNULL(totals.transactions, 0)) AS transactions,
  ROUND(SUM(IFNULL(totals.totalTransactionRevenue, 0)) / 1e6, 2) AS revenue_usd
FROM {TABLE}
WHERE {DATE_FILTER}
GROUP BY month
ORDER BY month""",
}


# --------------------------------------------------------------------------- #
# BigQuery helpers
# --------------------------------------------------------------------------- #
def estimate_gb(client, sql: str) -> float:
    """Dry run: BigQuery validates the SQL and reports bytes to be scanned,
    without running the query or charging anything."""
    from google.cloud import bigquery
    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True,
                                                               use_query_cache=False))
    return job.total_bytes_processed / 1024 ** 3


def run_query(client, sql: str, max_gb: float) -> pd.DataFrame:
    """Run the query with a hard cap on billed bytes (like bq_helper's
    query_to_pandas_safe)."""
    from google.cloud import bigquery
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * 1024 ** 3))
    return client.query(sql, job_config=cfg).to_dataframe()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save(fig, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_figures(res: dict[str, pd.DataFrame], out: Path) -> None:
    c = res["users_by_country"].head(10)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = range(len(c))
    axes[0].bar([i - 0.2 for i in x], c["users"], width=0.4, label="Users", color="indianred")
    axes[0].bar([i + 0.2 for i in x], c["new_users"], width=0.4, label="New users",
                color="steelblue")
    axes[0].set_xticks(list(x), c["country"], rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_title("Users and new users, top 10 countries")
    axes[1].bar(c["country"], c["conversion_rate_pct"], color="seagreen")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylabel("% of users who bought")
    axes[1].set_title("Conversion rate, top 10 countries by users")
    save(fig, out, "01_users_and_conversion_by_country.png")

    s = res["pageviews_by_subcontinent"].groupby("region")["total_pageviews"].sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    s.sort_values().plot.barh(ax=ax, color="slateblue")
    ax.set_title("Page views by region (Americas split by sub continent)")
    save(fig, out, "02_pageviews_by_region.png")

    for key, label, name in [("conversion_by_device", "device", "03_conversion_by_device.png"),
                             ("conversion_by_channel", "channel", "04_conversion_by_channel.png")]:
        d = res[key]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(d[label], d["session_conversion_pct"], color="darkorange")
        ax.set_ylabel("% of sessions with a purchase")
        ax.set_title(f"Session conversion rate by {label}")
        ax.tick_params(axis="x", rotation=30)
        save(fig, out, name)

    m = res["monthly_trend"]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(m["month"], m["sessions"], marker="o", label="Sessions")
    ax1.set_ylabel("Sessions")
    ax2 = ax1.twinx()
    ax2.plot(m["month"], m["revenue_usd"], marker="s", color="green", label="Revenue (USD)")
    ax2.set_ylabel("Revenue (USD)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_title("Monthly sessions and revenue")
    save(fig, out, "05_monthly_trend.png")

    try:
        import plotly.graph_objects as go
        full = res["users_by_country"]
        fig = go.Figure(go.Choropleth(locations=full["country"], locationmode="country names",
                                      z=full["conversion_rate_pct"], colorscale="Greens",
                                      colorbar_title="Conversion %"))
        fig.update_layout(title="Visitor conversion rate by country (top 15 by users)")
        fig.write_html(out / "06_conversion_choropleth.html")
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, help="your Google Cloud project id")
    parser.add_argument("--dry-run", action="store_true", help="only estimate scanned GB")
    parser.add_argument("--max-gb", type=float, default=1.0,
                        help="refuse to run any query that would bill more than this")
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)

    results = {}
    for name, sql in QUERIES.items():
        gb = estimate_gb(client, sql)
        print(f"\n=== {name}: will scan {gb:.3f} GB ===")
        if args.dry_run:
            continue
        df = run_query(client, sql, args.max_gb)
        results[name] = df
        print(df.to_string(index=False))
        args.results.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.results / f"{name}.csv", index=False)

    if results:
        make_figures(results, args.figures)
        print(f"\nTables saved to {args.results}/, figures saved to {args.figures}/")


if __name__ == "__main__":
    main()
