"""
export.py - Milestone 7: collect everything the website shows into one file,
site/data.json.

The website is fully static (plain HTML/CSS/JS in site/), so it can't query
DuckDB. Instead this script runs at the end of the pipeline and writes one
small JSON file with today's numbers: KPIs, the revenue forecast and its
backtest, pipeline by stage, the top deals (with their LLM briefings), hot
leads, customer segments, next best actions and model accuracy.

Only analytics results are exported - never data/sim_state (the simulator's
hidden truth) and never anything secret.

Usage (PowerShell):
  python export.py
"""

import json
import math
from datetime import date, datetime
from decimal import Decimal

import duckdb

from load import DB_PATH, ROOT

OUT_FILE = ROOT / "site" / "data.json"
BRIEFINGS_FILE = ROOT / "data" / "briefings" / "deal_briefings.json"
TOP_DEALS = 25
TOP_LEADS = 15
TOP_ACTIONS = 15

# Readable names for the lead-scoring features shown on the website
FEATURE_NAMES = {
    "responses_log": "Buyer responses", "responses_7d_log": "Buyer responses this week",
    "days_since_response": "Days since last buyer response", "meetings_log": "Meetings held",
    "touches_log": "Rep touches", "days_since_touch": "Days since last rep touch",
    "age_log": "Lead age", "is_existing_customer": "Already a customer",
    "job_title_missing": "Contact's job title missing",
}
CATEGORY_PREFIXES = {"size_band_": "Company size", "seniority_": "Contact seniority",
                     "industry_": "Industry", "source_": "Lead source", "region_": "Region"}


def plain(v):
    """DuckDB values -> JSON-friendly values (dates as ISO text, rounded floats)."""
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 4)
    return v


def rows(con, sql, params=None):
    cur = con.execute(sql, params or [])
    cols = [c[0] for c in cur.description]
    return [{c: plain(v) for c, v in zip(cols, r)} for r in cur.fetchall()]


def split_list(text):
    return [t for t in (text or "").split("; ") if t]


def feature_label(name):
    if name in FEATURE_NAMES:
        return FEATURE_NAMES[name]
    for prefix, label in CATEGORY_PREFIXES.items():
        if name.startswith(prefix):
            return f"{label}: {name.removeprefix(prefix)}"
    return name


def kpis(con):
    k = rows(con, """
        SELECT count(*) AS open_deals, sum(amount_usd) AS pipeline_usd,
               sum(amount_usd * rep_probability_pct / 100) AS rep_weighted_usd,
               sum(amount_usd * win_probability_pct / 100) AS model_weighted_usd
        FROM analytics.deal_scores""")[0]
    k |= rows(con, """
        SELECT count(*) FILTER (WHERE is_won) AS won_deals_90d,
               count(*) AS closed_deals_90d,
               coalesce(sum(amount_usd) FILTER (WHERE is_won), 0) AS bookings_90d_usd
        FROM clean.deals, (SELECT data_through AS d FROM raw.export_info)
        WHERE is_closed AND closed_date > d - 90""")[0]
    k |= rows(con, """
        SELECT count(*) AS open_leads, count(*) FILTER (WHERE grade = 'A') AS grade_a_leads
        FROM analytics.lead_scores""")[0]
    k["win_rate_90d"] = k["won_deals_90d"] / k["closed_deals_90d"] if k["closed_deals_90d"] else None
    return k


def forecast(con):
    today = rows(con, "SELECT * FROM analytics.revenue_forecast ORDER BY horizon_days")
    backtest, accuracy = {}, {}
    for h in (r["horizon_days"] for r in today):
        bt = rows(con, """
            SELECT forecast_date AS date, forecast_usd AS kairo_usd, actual_usd,
                   rep_weighted_usd, run_rate_usd
            FROM analytics.forecast_backtest WHERE horizon_days = ? ORDER BY forecast_date""", [h])
        backtest[str(h)] = bt
        accuracy[str(h)] = {
            method: sum(abs(r[f"{method}_usd"] - r["actual_usd"]) / r["actual_usd"] for r in bt) / len(bt)
            for method in ("kairo", "rep_weighted", "run_rate")} | {"forecasts": len(bt)}
    return {"today": today, "backtest": backtest, "accuracy": accuracy}


def deals(con):
    briefings = {}
    if BRIEFINGS_FILE.exists():
        briefings = {b["deal_id"]: b for b in json.loads(BRIEFINGS_FILE.read_text(encoding="utf-8"))}
    top = rows(con, f"""
        SELECT s.deal_id, s.deal_name, s.account_name, s.owner_name, s.deal_type, s.stage,
               s.amount_usd, s.expected_close_date, s.rep_probability_pct,
               s.win_probability_pct AS model_probability_pct,
               -- capped: winning within 30 days can't be likelier than winning at all
               round(least(f.p_won * 100, s.win_probability_pct), 1) AS p_won_30d_pct,
               s.risk_flags, s.top_reasons,
               n.action, n.reason, n.rep_vs_model, n.value_usd
        FROM analytics.next_best_actions n
        JOIN analytics.deal_scores s ON s.deal_id = n.object_id
        JOIN analytics.deal_close_forecast f ON f.deal_id = s.deal_id AND f.horizon_days = 30
        WHERE n.object_type = 'deal' AND n.category = 'revenue'
        ORDER BY n.priority DESC, s.deal_id LIMIT {TOP_DEALS}""")
    for d in top:
        d["risk_flags"] = split_list(d["risk_flags"])
        d["top_reasons"] = [{k: r[k] for k in ("factor", "effect", "detail")}
                            for r in json.loads(d["top_reasons"] or "[]")]
        b = briefings.get(d["deal_id"])
        # situation / why / action (current format); headline / risks / next_steps (older briefings)
        d["briefing"] = ({k: b[k] for k in ("situation", "why", "action", "headline", "risks",
                                            "next_steps", "generated_on", "model",
                                            "stale", "stale_reason") if k in b}
                         if b else None)
    return top


def leads(con):
    top = rows(con, f"""
        SELECT s.lead_id, a.account_name, s.source, s.industry, s.size_band, s.score, s.grade,
               s.top_reasons, r.name AS owner_name, s.days_since_created
        FROM analytics.lead_scores s
        JOIN clean.accounts a ON a.account_id = s.account_id
        JOIN clean.sales_reps r ON r.rep_id = s.owner_rep_id
        ORDER BY s.score DESC, s.lead_id LIMIT {TOP_LEADS}""")
    for lead in top:
        lead["top_reasons"] = split_list(lead["top_reasons"])
    grades = rows(con, "SELECT grade, count(*) AS leads FROM analytics.lead_scores GROUP BY grade ORDER BY grade")
    drivers = rows(con, "SELECT feature, effect FROM analytics.lead_score_drivers ORDER BY effect DESC")
    strongest = drivers[:6] + drivers[-6:]
    for d in strongest:
        d["label"] = feature_label(d["feature"])
    return {"top": top, "grades": grades, "drivers": strongest}


def actions(con):
    summary = rows(con, """
        SELECT action, category, count(*) AS count, sum(value_usd) AS value_usd
        FROM analytics.next_best_actions GROUP BY ALL
        ORDER BY category DESC, value_usd DESC, count DESC""")
    top = rows(con, f"""
        SELECT rank_overall, owner_name, action, object_type, account_name, value_usd, reason
        FROM analytics.next_best_actions WHERE category = 'revenue'
        ORDER BY rank_overall LIMIT {TOP_ACTIONS}""")
    orphaned = rows(con, """
        SELECT count(*) AS n FROM analytics.next_best_actions WHERE NOT owner_is_active""")[0]["n"]
    return {"summary": summary, "top": top, "owner_left_company": orphaned}


def models(con):
    out = {}
    for r in rows(con, "SELECT model, metric, value FROM analytics.model_metrics ORDER BY model, metric"):
        out.setdefault(r["model"], {})[r["metric"]] = json.loads(r["value"])
    return out


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    meta = rows(con, """
        SELECT data_through AS as_of,
               (SELECT min(created_date) FROM clean.deals) AS crm_start
        FROM raw.export_info""")[0]
    data = {
        "meta": meta | {"note": "All companies, people and deals are simulated. "
                                 "No real customer data is used."},
        "kpis": kpis(con),
        "forecast": forecast(con),
        "bookings_monthly": rows(con, """
            SELECT strftime(date_trunc('month', closed_date), '%Y-%m') AS month,
                   sum(amount_usd) AS won_usd, count(*) AS won_deals
            FROM clean.deals WHERE is_won GROUP BY 1 ORDER BY 1"""),
        "pipeline_by_stage": rows(con, """
            SELECT s.stage, count(*) AS deals, sum(s.amount_usd) AS amount_usd,
                   sum(s.amount_usd * s.rep_probability_pct / 100) AS rep_weighted_usd,
                   sum(s.amount_usd * s.win_probability_pct / 100) AS model_weighted_usd
            FROM analytics.deal_scores s JOIN clean.stages st USING (stage)
            GROUP BY s.stage, st.stage_order ORDER BY st.stage_order"""),
        "deals": deals(con),
        "leads": leads(con),
        "segments": rows(con, "SELECT * EXCLUDE (as_of_date) FROM analytics.segment_profiles ORDER BY segment_id"),
        "actions": actions(con),
        "models": models(con),
    }
    con.close()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT_FILE.write_text(text, encoding="utf-8")
    briefed = sum(d["briefing"] is not None for d in data["deals"])
    print(f"Wrote {OUT_FILE} ({len(text) / 1024:.0f} KB), data as of {meta['as_of']}")
    print(f"  {len(data['deals'])} deals ({briefed} with a briefing), {len(data['leads']['top'])} leads, "
          f"{len(data['segments'])} segments, {sum(len(v) for v in data['forecast']['backtest'].values())} "
          f"backtest forecasts")


if __name__ == "__main__":
    main()
