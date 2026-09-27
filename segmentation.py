"""
segmentation.py - Milestone 5a: group customers into segments that need
different treatment.

Every customer (account with at least one won deal) is described by how much
it has bought, how recently, how engaged it is right now, whether it has an
open deal, and its size. K-means clustering then groups similar customers.
Each segment gets a readable name from its profile (e.g. "High-value - quiet"),
so the names stay meaningful even when the clusters shift a little day to day.

N_SEGMENTS is fixed (not re-chosen every day) so segments stay stable for the
website. It was picked by looking at the silhouette score (how clearly the
groups separate, printed on every run) and at which split is most useful.

Writes to data/kairo.duckdb:
  analytics.customer_segments   one row per customer with its segment
  analytics.segment_profiles    what a typical customer in each segment looks like

Usage (PowerShell):
  python segmentation.py
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from load import DB_PATH
from ml_utils import save_table

N_SEGMENTS = 5
SIZE_ORDER = {"SMB": 1, "Mid-Market": 2, "Enterprise": 3}  # 'Unknown' -> middle (2)


def load_customers(con):
    return con.execute("""
        WITH dt AS (SELECT data_through AS d FROM raw.export_info),
        won AS (
            SELECT account_id, sum(amount_usd) AS lifetime_revenue_usd, count(*) AS won_deals,
                   max(closed_date) AS last_win_date
            FROM clean.deals WHERE is_won GROUP BY account_id
        ), act AS (
            SELECT account_id, count(*) AS activities_90d,
                   count(*) FILTER (WHERE is_buyer_response) AS buyer_responses_90d
            FROM clean.activities, dt WHERE activity_date > dt.d - 90 GROUP BY account_id
        ), pipeline AS (   -- open deals, valued with the model's win probability
            SELECT d.account_id, count(*) AS open_deals,
                   sum(coalesce(s.amount_usd, 0) * s.win_probability_pct / 100) AS open_pipeline_usd
            FROM analytics.deal_scores s JOIN clean.deals d USING (deal_id) GROUP BY d.account_id
        )
        SELECT a.account_id, a.account_name, a.industry, a.size_band, a.region, a.owner_rep_id,
               a.customer_since, w.lifetime_revenue_usd, w.won_deals,
               date_diff('day', a.customer_since, dt.d) AS customer_days,
               date_diff('day', w.last_win_date, dt.d) AS days_since_last_win,
               coalesce(act.activities_90d, 0) AS activities_90d,
               coalesce(act.buyer_responses_90d, 0) AS buyer_responses_90d,
               coalesce(p.open_deals, 0) AS open_deals,
               coalesce(p.open_pipeline_usd, 0) AS open_pipeline_usd
        FROM clean.accounts a
        JOIN won w USING (account_id)
        CROSS JOIN dt
        LEFT JOIN act USING (account_id)
        LEFT JOIN pipeline p USING (account_id)
        ORDER BY a.account_id
    """).df()


def features(df):
    """Numbers K-means compares. Money and counts are log-scaled so a few huge
    customers don't dominate; everything is then put on the same scale."""
    return pd.DataFrame({
        "revenue": np.log1p(df["lifetime_revenue_usd"]),
        "won_deals": df["won_deals"],
        "days_since_last_win": df["days_since_last_win"],
        "customer_days": df["customer_days"],
        "activity": np.log1p(df["activities_90d"]),
        "responses": np.log1p(df["buyer_responses_90d"]),
        "open_pipeline": np.log1p(df["open_pipeline_usd"]),
        "size": df["size_band"].map(SIZE_ORDER).fillna(2),
    })


def name_segment(profile, overall):
    """Readable name from what makes the segment's typical customer stand out."""
    if profile["open_deals_share"] >= 0.5:
        main = "Expanding"
    elif profile["won_deals_mean"] >= 1.5:
        main = "Repeat buyers"
    elif profile["days_since_last_win_median"] <= 90:
        main = "New customers"
    elif profile["revenue_median"] >= overall["lifetime_revenue_usd"].quantile(0.6):
        main = "High-value"
    elif profile["revenue_median"] <= overall["lifetime_revenue_usd"].quantile(0.4):
        main = "Small"
    else:
        main = "Core"
    engagement = "engaged" if profile["activities_90d_median"] >= 5 else "quiet"
    return f"{main} - {engagement}"


def main():
    con = duckdb.connect(str(DB_PATH))
    data_through = con.execute("SELECT data_through FROM raw.export_info").fetchone()[0]
    customers = load_customers(con)
    Z = StandardScaler().fit_transform(features(customers))

    print(f"Segmenting {len(customers)} customers as of {data_through}")
    print("  silhouette by number of segments (higher = more clearly separated groups):")
    for k in range(3, 8):
        labels = KMeans(k, n_init=20, random_state=42).fit_predict(Z)
        print(f"    {k} segments: {silhouette_score(Z, labels):.3f}" + ("   <- used" if k == N_SEGMENTS else ""))

    customers["cluster"] = KMeans(N_SEGMENTS, n_init=20, random_state=42).fit_predict(Z)
    profiles = customers.groupby("cluster").agg(
        customers=("account_id", "size"),
        revenue_median=("lifetime_revenue_usd", "median"),
        revenue_total=("lifetime_revenue_usd", "sum"),
        won_deals_mean=("won_deals", "mean"),
        days_since_last_win_median=("days_since_last_win", "median"),
        activities_90d_median=("activities_90d", "median"),
        open_deals_share=("open_deals", lambda s: (s > 0).mean()),
        enterprise_share=("size_band", lambda s: (s == "Enterprise").mean()),
    )
    profiles["segment_name"] = [name_segment(p, customers) for _, p in profiles.iterrows()]
    # Same name twice? Number them so they stay distinguishable.
    dup = profiles.groupby("segment_name").cumcount()
    profiles.loc[dup > 0, "segment_name"] += " (" + (dup[dup > 0] + 1).astype(str) + ")"
    # Stable ids: segment 1 = highest total revenue
    profiles = profiles.sort_values("revenue_total", ascending=False)
    profiles["segment_id"] = range(1, len(profiles) + 1)

    customers = customers.merge(profiles[["segment_id", "segment_name"]], left_on="cluster",
                                right_index=True).drop(columns="cluster")
    customers.insert(0, "as_of_date", data_through)
    save_table(con, "analytics.customer_segments", customers.sort_values(["segment_id", "account_id"]))
    out = profiles.reset_index(drop=True)
    out.insert(0, "as_of_date", data_through)
    save_table(con, "analytics.segment_profiles",
               out[["as_of_date", "segment_id", "segment_name"] + list(profiles.columns[:8])])

    print(f"\n{'segment':<28}{'customers':>10}{'median $':>10}{'total $':>12}{'deals':>7}"
          f"{'last win':>10}{'acts 90d':>9}{'open deal':>10}")
    for p in out.itertuples():
        print(f"{p.segment_id}. {p.segment_name:<25}{p.customers:>10}{p.revenue_median:>10,.0f}"
              f"{p.revenue_total:>12,.0f}{p.won_deals_mean:>7.1f}{p.days_since_last_win_median:>8.0f}d"
              f"{p.activities_90d_median:>9.0f}{p.open_deals_share:>10.0%}")
    print("(medians per segment; 'deals' = average won deals; 'last win' = days since last won deal)")
    con.close()


if __name__ == "__main__":
    main()
