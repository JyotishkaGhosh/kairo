"""
transform.py - clean the raw CRM tables and build daily deal snapshots.

Reads the `raw` schema made by load.py and writes, in data/kairo.duckdb:

  clean.*      tidy copies of every table: duplicate contacts merged, missing
               categories labelled 'Unknown' / 'Not specified', handy extra
               columns (dates, last activity, overdue flags).
  analytics.deal_daily_snapshots
               one row per open deal per day: what the deal looked like at
               the END of that day, using only information known on that day
               (point-in-time). Columns starting with `outcome_` are the
               exception: they tell how the deal finally ended and exist only
               as training labels - never use them as model inputs.
  analytics.lead_daily_snapshots
               the same for leads (used by lead_scoring.py).

Finally it runs data-quality checks and stops with an error if any fails.

Usage (PowerShell):
  python transform.py
"""

import duckdb

from load import DB_PATH

DATA_THROUGH = "(SELECT data_through FROM raw.export_info)"

# "Buyer response" = the customer actually engaged, not just the rep reaching out.
BUYER_RESPONSE = "(direction = 'inbound' OR outcome IN ('connected', 'held'))"

STEPS = {
    "clean.stages": """
        CREATE OR REPLACE TABLE clean.stages AS
        SELECT * FROM (VALUES
            ('Prospecting', 1, false), ('Qualified', 2, false), ('Demo', 3, false),
            ('Proposal', 4, false), ('Negotiation', 5, false),
            ('Closed Won', 6, true), ('Closed Lost', 6, true)
        ) AS t(stage, stage_order, is_closed_stage)
    """,

    # The same person entered twice at the same company -> keep the oldest
    # record as the "master" and point every duplicate at it.
    "clean.contact_id_map": """
        CREATE OR REPLACE TABLE clean.contact_id_map AS
        SELECT contact_id,
               first_value(contact_id) OVER (
                   PARTITION BY account_id, lower(trim(first_name)), lower(trim(last_name))
                   ORDER BY created_at, contact_id) AS master_contact_id
        FROM raw.contacts
    """,

    "clean.contacts": """
        CREATE OR REPLACE TABLE clean.contacts AS
        WITH merged AS (   -- fill gaps in the master record from its duplicates
            SELECT m.master_contact_id, count(*) - 1 AS duplicates_merged,
                   max(c.phone) AS phone, max(c.job_title) AS job_title,
                   max(c.seniority) AS seniority
            FROM raw.contacts c JOIN clean.contact_id_map m USING (contact_id)
            GROUP BY m.master_contact_id
        )
        SELECT c.contact_id, c.account_id,
               trim(c.first_name) AS first_name, trim(c.last_name) AS last_name,
               lower(trim(c.email)) AS email,
               coalesce(c.phone, g.phone) AS phone,
               coalesce(c.job_title, g.job_title) AS job_title,
               coalesce(c.seniority, g.seniority, 'Unknown') AS seniority,
               c.created_at, g.duplicates_merged
        FROM raw.contacts c
        JOIN merged g ON g.master_contact_id = c.contact_id
    """,

    "clean.sales_reps": """
        CREATE OR REPLACE TABLE clean.sales_reps AS
        SELECT *, date_diff('day', hire_date, coalesce(termination_date, {data_through})) AS tenure_days
        FROM raw.sales_reps
    """,

    "clean.accounts": """
        CREATE OR REPLACE TABLE clean.accounts AS
        SELECT a.account_id, trim(a.account_name) AS account_name, lower(trim(a.domain)) AS domain,
               coalesce(a.industry, 'Unknown') AS industry,
               a.employee_count,
               coalesce(a.size_band,
                        CASE WHEN a.employee_count < 200 THEN 'SMB'
                             WHEN a.employee_count < 2000 THEN 'Mid-Market'
                             WHEN a.employee_count IS NOT NULL THEN 'Enterprise' END,
                        'Unknown') AS size_band,
               a.region, a.country, a.annual_revenue_usd, a.owner_rep_id, a.created_at,
               w.customer_since,
               w.customer_since IS NOT NULL AS is_customer
        FROM raw.accounts a
        LEFT JOIN (SELECT account_id, min(CAST(closed_at AS DATE)) AS customer_since
                   FROM raw.deals WHERE is_won GROUP BY account_id) w USING (account_id)
    """,

    "clean.activities": """
        CREATE OR REPLACE TABLE clean.activities AS
        SELECT a.activity_id, a.activity_at, CAST(a.activity_at AS DATE) AS activity_date,
               a.activity_type, a.direction, a.outcome, a.duration_min, a.rep_id,
               m.master_contact_id AS contact_id, a.account_id, a.lead_id, a.deal_id,
               {buyer_response} AS is_buyer_response
        FROM raw.activities a
        LEFT JOIN clean.contact_id_map m USING (contact_id)
    """,

    "clean.leads": """
        CREATE OR REPLACE TABLE clean.leads AS
        WITH act AS (
            SELECT lead_id, max(activity_date) AS last_activity_date,
                   count(*) AS activity_count,
                   count(*) FILTER (WHERE is_buyer_response) AS buyer_responses
            FROM clean.activities WHERE lead_id IS NOT NULL GROUP BY lead_id
        )
        SELECT l.lead_id, l.created_at, CAST(l.created_at AS DATE) AS created_date,
               l.source, m.master_contact_id AS contact_id, l.account_id, l.owner_rep_id,
               l.status, l.status_changed_at,
               CASE WHEN l.status = 'Disqualified'
                    THEN coalesce(l.disqualify_reason, 'Not specified') END AS disqualify_reason,
               l.converted_deal_id,
               l.status IN ('New', 'Working') AS is_open,
               coalesce(act.activity_count, 0) AS activity_count,
               coalesce(act.buyer_responses, 0) AS buyer_responses,
               act.last_activity_date,
               -- open, but nobody has touched it for 30+ days: forgotten
               l.status IN ('New', 'Working')
                   AND coalesce(act.last_activity_date, CAST(l.created_at AS DATE)) < {data_through} - 30
                   AS is_stale
        FROM raw.leads l
        LEFT JOIN clean.contact_id_map m USING (contact_id)
        LEFT JOIN act USING (lead_id)
    """,

    "clean.deal_stage_history": """
        CREATE OR REPLACE TABLE clean.deal_stage_history AS
        SELECT *, CAST(changed_at AS DATE) AS change_date
        FROM raw.deal_stage_history
    """,

    "clean.deals": """
        CREATE OR REPLACE TABLE clean.deals AS
        WITH act AS (
            SELECT deal_id, max(activity_date) AS last_activity_date,
                   max(activity_date) FILTER (WHERE is_buyer_response) AS last_buyer_response_date
            FROM clean.activities WHERE deal_id IS NOT NULL GROUP BY deal_id
        ), pushes AS (
            SELECT deal_id, count(*) AS close_date_pushes
            FROM raw.deal_stage_history WHERE change_type = 'close_date_change' GROUP BY deal_id
        )
        SELECT d.deal_id, d.deal_name, d.account_id,
               m.master_contact_id AS primary_contact_id, d.lead_id, d.owner_rep_id,
               d.deal_type,
               coalesce(d.source, CASE WHEN d.deal_type = 'Expansion'
                                       THEN 'existing_customer' ELSE 'unknown' END) AS source,
               d.product_tier, d.stage, s.stage_order,
               d.amount_usd, d.probability_pct, d.expected_close_date,
               d.created_at, CAST(d.created_at AS DATE) AS created_date,
               d.stage_changed_at, d.closed_at, CAST(d.closed_at AS DATE) AS closed_date,
               d.is_closed, d.is_won,
               CASE WHEN d.is_closed AND NOT d.is_won
                    THEN coalesce(d.lost_reason, 'Not specified') END AS lost_reason,
               coalesce(p.close_date_pushes, 0) AS close_date_pushes,
               act.last_activity_date, act.last_buyer_response_date,
               NOT d.is_closed AND d.expected_close_date < {data_through} AS is_close_date_past
        FROM raw.deals d
        JOIN clean.stages s USING (stage)
        LEFT JOIN clean.contact_id_map m ON m.contact_id = d.primary_contact_id
        LEFT JOIN act USING (deal_id)
        LEFT JOIN pushes p USING (deal_id)
    """,

    # ------------------------------------------------------------------
    # Point-in-time snapshots. For every day a deal was open we rebuild its
    # state from the change log (deal_stage_history) and count activity up
    # to that day - never later. This is what lets us train and backtest
    # models "as if it were that day", without peeking into the future.
    # ------------------------------------------------------------------
    "analytics.deal_daily_snapshots": """
        CREATE OR REPLACE TABLE analytics.deal_daily_snapshots AS
        WITH hist AS (
            -- running values along each deal's change log (history_id = log order)
            SELECT deal_id, history_id, change_date, to_stage AS stage, amount_usd,
                   probability_pct, expected_close_date, owner_rep_id,
                   count(*) FILTER (WHERE change_type = 'close_date_change') OVER w AS close_date_pushes,
                   last_value(CASE WHEN change_type IN ('created', 'stage_change') THEN change_date END
                              IGNORE NULLS) OVER w AS stage_entered_date
            FROM clean.deal_stage_history
            WINDOW w AS (PARTITION BY deal_id ORDER BY history_id ROWS UNBOUNDED PRECEDING)
        ), end_of_day AS (
            -- the deal's state at the end of each day it changed
            SELECT * FROM hist
            QUALIFY row_number() OVER (PARTITION BY deal_id, change_date ORDER BY history_id DESC) = 1
        ), spine AS (
            -- every day the deal was open at end of day: created .. day before close
            SELECT deal_id,
                   CAST(unnest(generate_series(created_date,
                                               coalesce(closed_date - 1, {data_through}),
                                               INTERVAL 1 DAY)) AS DATE) AS snapshot_date
            FROM clean.deals
        ), act_daily AS (
            SELECT deal_id, activity_date,
                   count(*) AS n_activities,
                   count(*) FILTER (WHERE is_buyer_response) AS n_buyer_responses,
                   count(*) FILTER (WHERE activity_type = 'meeting' AND outcome = 'held') AS n_meetings
            FROM clean.activities WHERE deal_id IS NOT NULL
            GROUP BY ALL
        ), act_running AS (
            -- the spine has one row per day, so "6 preceding rows" = the last 7 days
            SELECT s.deal_id, s.snapshot_date,
                   CAST(sum(coalesce(a.n_activities, 0)) OVER w7 AS INTEGER) AS activities_7d,
                   CAST(sum(coalesce(a.n_activities, 0)) OVER w30 AS INTEGER) AS activities_30d,
                   CAST(sum(coalesce(a.n_activities, 0)) OVER w_all AS INTEGER) AS activities_total,
                   CAST(sum(coalesce(a.n_buyer_responses, 0)) OVER w30 AS INTEGER) AS buyer_responses_30d,
                   CAST(sum(coalesce(a.n_buyer_responses, 0)) OVER w_all AS INTEGER) AS buyer_responses_total,
                   CAST(sum(coalesce(a.n_meetings, 0)) OVER w_all AS INTEGER) AS meetings_total,
                   max(a.activity_date) OVER w_all AS last_activity_date,
                   max(CASE WHEN a.n_buyer_responses > 0 THEN a.activity_date END) OVER w_all
                       AS last_buyer_response_date
            FROM spine s
            LEFT JOIN act_daily a ON a.deal_id = s.deal_id AND a.activity_date = s.snapshot_date
            WINDOW w7 AS (PARTITION BY s.deal_id ORDER BY s.snapshot_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
                   w30 AS (PARTITION BY s.deal_id ORDER BY s.snapshot_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
                   w_all AS (PARTITION BY s.deal_id ORDER BY s.snapshot_date ROWS UNBOUNDED PRECEDING)
        )
        SELECT r.snapshot_date, r.deal_id,
               -- facts that never change
               d.account_id, d.deal_type, d.source, d.product_tier,
               a.industry, a.size_band, a.region,
               -- the deal as recorded in the CRM on snapshot_date
               h.owner_rep_id, h.stage, s.stage_order, h.amount_usd, h.probability_pct,
               h.expected_close_date, h.close_date_pushes,
               date_diff('day', d.created_date, r.snapshot_date) AS days_since_created,
               date_diff('day', h.stage_entered_date, r.snapshot_date) AS days_in_stage,
               date_diff('day', r.snapshot_date, h.expected_close_date) AS days_to_expected_close,
               h.expected_close_date < r.snapshot_date AS is_close_date_past,
               -- engagement up to and including snapshot_date
               r.activities_7d, r.activities_30d, r.activities_total,
               r.buyer_responses_30d, r.buyer_responses_total, r.meetings_total,
               date_diff('day', r.last_activity_date, r.snapshot_date) AS days_since_last_activity,
               date_diff('day', r.last_buyer_response_date, r.snapshot_date) AS days_since_buyer_response,
               -- LABELS ONLY (future information): how the deal ended
               d.is_closed AS outcome_is_closed,
               d.is_won AS outcome_is_won,
               d.closed_date AS outcome_closed_date,
               CASE WHEN d.is_won THEN d.amount_usd END AS outcome_won_amount_usd,
               date_diff('day', r.snapshot_date, d.closed_date) AS outcome_days_until_close
        FROM act_running r
        ASOF JOIN end_of_day h ON h.deal_id = r.deal_id AND h.change_date <= r.snapshot_date
        JOIN clean.deals d ON d.deal_id = r.deal_id
        JOIN clean.accounts a ON a.account_id = d.account_id
        JOIN clean.stages s ON s.stage = h.stage
        ORDER BY r.snapshot_date, r.deal_id
    """,

    # Same idea for leads: one row per open lead per day, engagement counted
    # up to that day only. Used to train and apply the lead scoring model.
    "analytics.lead_daily_snapshots": """
        CREATE OR REPLACE TABLE analytics.lead_daily_snapshots AS
        WITH leads AS (
            SELECT *, CASE WHEN NOT is_open THEN CAST(status_changed_at AS DATE) END AS resolved_date
            FROM clean.leads
        ), spine AS (
            -- every day the lead was open at end of day: created .. day before resolved
            SELECT lead_id,
                   CAST(unnest(generate_series(created_date,
                                               coalesce(resolved_date - 1, {data_through}),
                                               INTERVAL 1 DAY)) AS DATE) AS snapshot_date
            FROM leads
        ), act_daily AS (
            SELECT lead_id, activity_date,
                   count(*) FILTER (WHERE direction = 'outbound') AS n_touches,
                   count(*) FILTER (WHERE is_buyer_response) AS n_responses,
                   count(*) FILTER (WHERE activity_type = 'meeting' AND outcome = 'held') AS n_meetings
            FROM clean.activities WHERE lead_id IS NOT NULL
            GROUP BY ALL
        ), act_running AS (
            SELECT s.lead_id, s.snapshot_date,
                   CAST(sum(coalesce(a.n_touches, 0)) OVER w_all AS INTEGER) AS touches_total,
                   CAST(sum(coalesce(a.n_responses, 0)) OVER w_all AS INTEGER) AS buyer_responses_total,
                   CAST(sum(coalesce(a.n_responses, 0)) OVER w7 AS INTEGER) AS buyer_responses_7d,
                   CAST(sum(coalesce(a.n_meetings, 0)) OVER w_all AS INTEGER) AS meetings_total,
                   max(CASE WHEN a.n_touches > 0 THEN a.activity_date END) OVER w_all AS last_touch_date,
                   max(CASE WHEN a.n_responses > 0 THEN a.activity_date END) OVER w_all AS last_response_date
            FROM spine s
            LEFT JOIN act_daily a ON a.lead_id = s.lead_id AND a.activity_date = s.snapshot_date
            WINDOW w7 AS (PARTITION BY s.lead_id ORDER BY s.snapshot_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
                   w_all AS (PARTITION BY s.lead_id ORDER BY s.snapshot_date ROWS UNBOUNDED PRECEDING)
        )
        -- no owner_rep_id here: lead reassignments aren't logged, so the owner on a
        -- past day is unknown (use clean.leads for the current owner)
        SELECT r.snapshot_date, r.lead_id, l.account_id, l.contact_id, l.source, a.industry, a.size_band, a.region, c.seniority,
               c.job_title IS NULL AS job_title_missing,
               coalesce(a.customer_since <= r.snapshot_date, false) AS is_existing_customer,
               date_diff('day', l.created_date, r.snapshot_date) AS days_since_created,
               r.touches_total, r.buyer_responses_total, r.buyer_responses_7d, r.meetings_total,
               date_diff('day', r.last_touch_date, r.snapshot_date) AS days_since_last_touch,
               date_diff('day', r.last_response_date, r.snapshot_date) AS days_since_buyer_response,
               -- LABELS ONLY (future information): how the lead ended
               NOT l.is_open AS outcome_is_resolved,
               l.status = 'Converted' AS outcome_converted,
               l.resolved_date AS outcome_resolved_date
        FROM act_running r
        JOIN leads l ON l.lead_id = r.lead_id
        JOIN clean.accounts a ON a.account_id = l.account_id
        JOIN clean.contacts c ON c.contact_id = l.contact_id
        ORDER BY r.snapshot_date, r.lead_id
    """,
}

# Each check returns the number of BAD rows; anything above 0 fails the run.
CHECKS = {
    **{f"{table}: {key} is unique and not null": f"""
         SELECT count(*) - count(DISTINCT {key}) + count(*) FILTER (WHERE {key} IS NULL)
         FROM clean.{table}"""
       for table, key in [("sales_reps", "rep_id"), ("accounts", "account_id"),
                          ("contacts", "contact_id"), ("leads", "lead_id"), ("deals", "deal_id"),
                          ("deal_stage_history", "history_id"), ("activities", "activity_id")]},
    **{f"{child}.{col} points to an existing {parent}": f"""
         SELECT count(*) FROM clean.{child} c
         WHERE c.{col} IS NOT NULL AND c.{col} NOT IN (SELECT {parent_key} FROM clean.{parent})"""
       for child, col, parent, parent_key in [
           ("contacts", "account_id", "accounts", "account_id"),
           ("leads", "contact_id", "contacts", "contact_id"),
           ("leads", "account_id", "accounts", "account_id"),
           ("leads", "converted_deal_id", "deals", "deal_id"),
           ("deals", "account_id", "accounts", "account_id"),
           ("deals", "primary_contact_id", "contacts", "contact_id"),
           ("deals", "owner_rep_id", "sales_reps", "rep_id"),
           ("activities", "contact_id", "contacts", "contact_id"),
           ("activities", "deal_id", "deals", "deal_id"),
           ("activities", "lead_id", "leads", "lead_id"),
           ("deal_stage_history", "deal_id", "deals", "deal_id")]},
    "nothing is dated after the export date": f"""
        SELECT (SELECT count(*) FROM clean.activities WHERE activity_date > {DATA_THROUGH})
             + (SELECT count(*) FROM clean.deal_stage_history WHERE change_date > {DATA_THROUGH})
             + (SELECT count(*) FROM clean.leads WHERE created_date > {DATA_THROUGH})""",
    "change log rebuilds every deal's current state": """
        WITH last AS (
            SELECT * FROM clean.deal_stage_history
            QUALIFY row_number() OVER (PARTITION BY deal_id ORDER BY history_id DESC) = 1)
        SELECT count(*) FROM clean.deals d LEFT JOIN last l USING (deal_id)
        WHERE l.deal_id IS NULL
           OR l.to_stage IS DISTINCT FROM d.stage
           OR l.amount_usd IS DISTINCT FROM d.amount_usd
           OR l.probability_pct IS DISTINCT FROM d.probability_pct
           OR l.expected_close_date IS DISTINCT FROM d.expected_close_date
           OR l.owner_rep_id IS DISTINCT FROM d.owner_rep_id""",
    "snapshots: one row per deal per day": """
        SELECT count(*) - count(DISTINCT (deal_id, snapshot_date)) FROM analytics.deal_daily_snapshots""",
    "snapshots: only open stages, only while the deal was open": f"""
        SELECT count(*) FROM analytics.deal_daily_snapshots
        WHERE stage IN ('Closed Won', 'Closed Lost')
           OR snapshot_date > {DATA_THROUGH}
           OR (outcome_closed_date IS NOT NULL AND snapshot_date >= outcome_closed_date)""",
    "snapshots: latest day matches today's open pipeline": f"""
        WITH latest AS (SELECT * FROM analytics.deal_daily_snapshots
                        WHERE snapshot_date = {DATA_THROUGH}),
             open_now AS (SELECT * FROM clean.deals WHERE NOT is_closed)
        SELECT count(*) FROM open_now o FULL JOIN latest l USING (deal_id)
        WHERE o.deal_id IS NULL OR l.deal_id IS NULL
           OR o.stage <> l.stage OR o.amount_usd IS DISTINCT FROM l.amount_usd
           OR o.expected_close_date <> l.expected_close_date""",
    "lead snapshots: one row per lead per day": """
        SELECT count(*) - count(DISTINCT (lead_id, snapshot_date)) FROM analytics.lead_daily_snapshots""",
    "lead snapshots: only while the lead was open": f"""
        SELECT count(*) FROM analytics.lead_daily_snapshots
        WHERE snapshot_date > {DATA_THROUGH}
           OR (outcome_resolved_date IS NOT NULL AND snapshot_date >= outcome_resolved_date)""",
    "lead snapshots: latest day matches today's open leads": f"""
        SELECT count(*) FROM (SELECT lead_id FROM clean.leads WHERE is_open) o
        FULL JOIN (SELECT lead_id FROM analytics.lead_daily_snapshots
                   WHERE snapshot_date = {DATA_THROUGH}) l USING (lead_id)
        WHERE o.lead_id IS NULL OR l.lead_id IS NULL""",
}

QUALITY_REPORT = {
    "duplicate contact records merged": "SELECT sum(duplicates_merged) FROM clean.contacts",
    "accounts with unknown industry": "SELECT count(*) FROM clean.accounts WHERE industry = 'Unknown'",
    "accounts with unknown size": "SELECT count(*) FROM clean.accounts WHERE size_band = 'Unknown'",
    "open leads nobody touched in 30+ days": "SELECT count(*) FROM clean.leads WHERE is_stale",
    "open deals with no amount": "SELECT count(*) FROM clean.deals WHERE NOT is_closed AND amount_usd IS NULL",
    "open deals with a close date in the past": "SELECT count(*) FROM clean.deals WHERE is_close_date_past",
    "open deals with no buyer response for 21+ days": f"""
        SELECT count(*) FROM clean.deals
        WHERE NOT is_closed
          AND coalesce(last_buyer_response_date, created_date) < {DATA_THROUGH} - 21""",
    "lost deals with no lost reason": "SELECT count(*) FROM clean.deals WHERE lost_reason = 'Not specified'",
}


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS clean")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    data_through = con.execute(f"SELECT {DATA_THROUGH}").fetchone()[0]
    print(f"Transforming CRM data through {data_through}\n")

    print(f"{'table':<38}{'rows':>10}")
    for table, sql in STEPS.items():
        con.execute(sql.format(data_through=DATA_THROUGH, buyer_response=BUYER_RESPONSE))
        rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table:<38}{rows:>10,}")

    print("\nData checks")
    failed = 0
    for name, sql in CHECKS.items():
        bad = con.execute(sql).fetchone()[0]
        failed += bad > 0
        print(f"  {'PASS' if bad == 0 else 'FAIL'}  {name}" + (f"  ({bad} bad rows)" if bad else ""))

    print("\nData quality (known messiness, handled or flagged)")
    for name, sql in QUALITY_REPORT.items():
        print(f"  {con.execute(sql).fetchone()[0]:>6,}  {name}")

    print("\nOpen pipeline on the last day of recent months (rebuilt from snapshots)")
    print(con.execute(f"""
        SELECT snapshot_date AS as_of, count(*) AS open_deals,
               round(sum(amount_usd)) AS pipeline_usd,
               round(sum(amount_usd * probability_pct / 100)) AS rep_weighted_usd,
               count(*) FILTER (WHERE is_close_date_past) AS overdue_close_dates
        FROM analytics.deal_daily_snapshots
        WHERE snapshot_date = last_day(snapshot_date) OR snapshot_date = {DATA_THROUGH}
        GROUP BY snapshot_date ORDER BY snapshot_date DESC LIMIT 6
    """).df().to_string(index=False))
    con.close()

    if failed:
        raise SystemExit(f"\n{failed} data check(s) FAILED - see above.")
    print("\nAll data checks passed.")


if __name__ == "__main__":
    main()
