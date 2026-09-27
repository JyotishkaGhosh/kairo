"""
next_best_action.py - Milestone 5b: what should each rep do next?

Gives every open lead, open deal and customer ONE recommended action, using
the outputs of the earlier models:
  leads      lead score (lead_scoring.py)
  deals      win probability, P(won within 30 days), risk signals
             (win_probability.py, revenue_forecast.py)
  customers  segment (segmentation.py)

Two kinds of actions:
  revenue  actions that can win money. Ranked by
           priority = value at stake ($ x model probability) x urgency
  hygiene  keeping the CRM honest (close dead deals, drop stale leads, fix
           dates). No $ value of their own, but they make the forecast and
           everyone's pipeline trustworthy.

Honest limitation: the playbook is a set of transparent rules on top of the
models. It ranks WHERE attention is worth the most; it does not predict how
much an action changes the outcome. Measuring that needs real experiments
(and in this simulation, buyers do not react to reps' actions).

Writes to data/kairo.duckdb:
  analytics.next_best_actions   one row per lead / deal / customer

Usage (PowerShell):
  python next_best_action.py
"""

import duckdb
import pandas as pd

from load import DB_PATH
from ml_utils import plural, save_table

COMPLETE_AFTER_DAYS = 200  # deals this old have a known outcome (see win_probability.py)
URGENCY = {  # multiplies the $ value at stake
    "Push to close this month": 1.5,
    "Call today - the buyer is responding": 1.5,
    "Re-engage the buyer": 1.3,
    "Reset the overdue close date": 1.2,
    "Agree a close plan with the buyer": 1.2,
    "Unstick the deal": 1.1,
    "Follow up this week": 1.0,
    "Re-qualify the deal": 1.0,
    "Pitch an expansion": 1.0,
    "Strengthen the deal": 1.0,
    "Qualify the opportunity": 0.9,
    "Keep momentum": 0.8,
    "Decide if it is worth pursuing": 0.6,
    "Add to nurture": 0.5,
}
EARLY_STAGES = ("Prospecting", "Qualified")
LOW_WIN_PCT = 10          # below this model win probability a deal needs a go / no-go decision
BUYER_SILENT_DAYS = 21    # same threshold as the "no buyer response" risk flag
REP_QUIET_DAYS = 14       # the rep hasn't logged any activity for this long
REP_OPTIMISM_GAP = 25     # same as the risk flag in win_probability.py
PUSH_TO_CLOSE_P30 = 0.6   # "push to close" only if the deal is likely (60%+) to be won within 30 days
ON_TRACK_MIN_WIN_PCT = 50  # "on track" needs at least this model win probability ...
ON_TRACK_MAX_GAP = 15      # ... and the rep no more than this many points above the model


def history(con):
    """Typical outcomes from past deals, used to put a $ value on leads and customers."""
    h = con.execute(f"""
        WITH d AS (SELECT d.*, a.size_band FROM clean.deals d JOIN clean.accounts a USING (account_id)),
        dt AS (SELECT data_through FROM raw.export_info)
        SELECT
            (SELECT avg(is_won::int) FROM d, dt WHERE deal_type = 'New Business'
                AND created_date <= data_through - {COMPLETE_AFTER_DAYS}) AS new_business_win_rate,
            (SELECT avg(is_won::int) FROM d WHERE deal_type = 'Expansion' AND is_closed) AS expansion_win_rate,
            (SELECT median(amount_usd) FROM d WHERE deal_type = 'New Business' AND is_won) AS nb_median,
            (SELECT median(amount_usd) FROM d WHERE deal_type = 'Expansion' AND is_won) AS exp_median
    """).df().iloc[0].to_dict()
    by_size = con.execute("""
        SELECT a.size_band, median(d.amount_usd) AS median_won
        FROM clean.deals d JOIN clean.accounts a USING (account_id)
        WHERE d.is_won AND d.deal_type = 'New Business' GROUP BY a.size_band
    """).df().set_index("size_band")["median_won"].to_dict()
    h["nb_median_by_size"] = by_size
    # Expansion deals are few, so scale the overall expansion median by company size
    h["exp_median_by_size"] = {s: h["exp_median"] * v / h["nb_median"] for s, v in by_size.items()}
    return h


def lead_actions(con, h):
    leads = con.execute("""
        SELECT s.lead_id, s.account_id, a.account_name, l.owner_rep_id, s.score, s.grade,
               s.top_reasons, s.size_band, l.is_stale,
               coalesce(snap.days_since_buyer_response, snap.days_since_created) AS days_since_response
        FROM analytics.lead_scores s
        JOIN clean.leads l USING (lead_id)
        JOIN clean.accounts a ON a.account_id = s.account_id
        JOIN analytics.lead_daily_snapshots snap
            ON snap.lead_id = s.lead_id AND snap.snapshot_date = s.as_of_date
    """).df()
    rows = []
    for r in leads.itertuples():
        # value = P(becomes a deal) x P(deal is won) x typical first deal for this company size
        value = (r.score / 100 * h["new_business_win_rate"]
                 * h["nb_median_by_size"].get(r.size_band, h["nb_median"]))
        if r.is_stale:
            action, category, value = "Disqualify or move to nurture", "hygiene", 0.0
            reason = "open lead, nobody has touched it for 30+ days"
        elif r.grade == "A" and r.days_since_response <= 7:
            action, category = "Call today - the buyer is responding", "revenue"
            reason = f"lead score {r.score:.0f} (A): {r.top_reasons}"
        elif r.grade in ("A", "B"):
            action, category = "Follow up this week", "revenue"
            reason = f"lead score {r.score:.0f} ({r.grade}): {r.top_reasons}"
        else:
            action, category = "Add to nurture", "revenue"
            reason = f"low lead score {r.score:.0f} ({r.grade}): {r.top_reasons}"
        rows.append({"object_type": "lead", "object_id": r.lead_id, "account_id": r.account_id,
                     "account_name": r.account_name, "owner_rep_id": r.owner_rep_id,
                     "action": action, "category": category, "reason": reason, "value_usd": value})
    return rows


def deal_actions(con):
    deals = con.execute("""
        SELECT s.deal_id, d.account_id, s.account_name, s.deal_name, s.owner_rep_id, s.stage,
               s.win_probability_pct, s.rep_probability_pct, s.risk_flags,
               f90.expected_amount,
               -- two separate models: winning within 30 days can't be likelier than winning at all
               least(f30.p_won, s.win_probability_pct / 100) AS p_won_30d,
               coalesce(snap.days_since_buyer_response, snap.days_since_created) AS days_since_response,
               coalesce(snap.days_since_last_activity, snap.days_since_created) AS days_since_activity,
               snap.is_close_date_past, snap.close_date_pushes
        FROM analytics.deal_scores s
        JOIN clean.deals d USING (deal_id)
        JOIN analytics.deal_close_forecast f90 ON f90.deal_id = s.deal_id AND f90.horizon_days = 90
        JOIN analytics.deal_close_forecast f30 ON f30.deal_id = s.deal_id AND f30.horizon_days = 30
        JOIN analytics.deal_daily_snapshots snap
            ON snap.deal_id = s.deal_id AND snap.snapshot_date = s.as_of_date
    """).df()
    rows = []
    for r in deals.itertuples():
        p = r.win_probability_pct / 100
        value = p * r.expected_amount  # expected revenue from this deal
        rep_vs_model = f"rep says {r.rep_probability_pct}%, model says {r.win_probability_pct:.0f}%"
        gap = r.rep_probability_pct - r.win_probability_pct
        category = "revenue"
        stuck = next((f for f in r.risk_flags.split("; ") if f.startswith("stuck in")), None)
        # First matching rule wins. Problems that block the deal come before "push to close":
        # a deal that is closing soon but keeps slipping needs a close plan, not more pushing.
        if r.days_since_response >= 90 or r.win_probability_pct < 2:
            action, category = "Close as lost (clean up pipeline)", "hygiene"
            reason = (f"buyer silent for {plural(r.days_since_response, 'day')}, model {r.win_probability_pct:.1f}%; "
                      f"${r.expected_amount:,.0f} is inflating the pipeline")
            value = 0.0
        elif r.days_since_response >= BUYER_SILENT_DAYS:
            action = "Re-engage the buyer"
            reason = f"no buyer response for {plural(r.days_since_response, 'day')}"
        elif r.is_close_date_past:
            action = "Reset the overdue close date"
            reason = "the expected close date has passed; agree a realistic new date with the buyer"
        elif r.close_date_pushes >= 2:
            action = "Agree a close plan with the buyer"
            reason = (f"close date moved {plural(r.close_date_pushes, 'time')}; "
                      "agree the remaining steps and dates in writing")
        elif gap >= REP_OPTIMISM_GAP:
            action = "Re-qualify the deal"
            reason = "rep is far more optimistic than the model; check budget, decision maker and timeline"
        elif r.p_won_30d >= PUSH_TO_CLOSE_P30:
            action = "Push to close this month"
            reason = f"{r.p_won_30d:.0%} chance to be won within 30 days"
        elif stuck:
            action = "Unstick the deal"
            reason = f"{stuck}; agree a concrete next step or bring in a senior sponsor"
        elif r.days_since_activity >= REP_QUIET_DAYS:
            action = "Follow up this week"
            reason = f"no activity logged for {plural(r.days_since_activity, 'day')}"
        elif r.win_probability_pct < LOW_WIN_PCT:
            action = "Decide if it is worth pursuing"
            reason = f"only {r.win_probability_pct:.0f}% model win probability; qualify it out or find a sponsor"
        elif r.stage in EARLY_STAGES:
            action = "Qualify the opportunity"
            reason = "early stage; confirm budget, decision maker and timeline to move it forward"
        elif r.win_probability_pct >= ON_TRACK_MIN_WIN_PCT and gap < ON_TRACK_MAX_GAP:
            action = "Keep momentum"
            reason = "on track; book the next meeting"
        else:  # no single problem, but not safe enough to call "on track"
            action = "Strengthen the deal"
            reason = (f"rep is {gap:.0f} points above the model; " if gap >= ON_TRACK_MAX_GAP else
                      f"only {r.win_probability_pct:.0f}% model win probability; ")
            reason += "confirm the champion, budget and decision process"
        rows.append({"object_type": "deal", "object_id": r.deal_id, "account_id": r.account_id,
                     "account_name": r.account_name, "owner_rep_id": r.owner_rep_id,
                     "action": action, "category": category, "reason": reason,
                     "rep_vs_model": rep_vs_model, "value_usd": value})
    return rows


def customer_actions(con, h):
    customers = con.execute("""
        SELECT account_id, account_name, owner_rep_id, size_band, segment_name,
               lifetime_revenue_usd, customer_days, open_deals
        FROM analytics.customer_segments
    """).df()
    rows = []
    for r in customers.itertuples():
        if r.open_deals > 0:
            continue  # the open deal already has its own action
        if r.customer_days < 90:
            action, category, value = "Onboarding check-in", "hygiene", 0.0
            reason = f"new customer ({r.customer_days} days); make sure they are live and happy"
        else:
            action, category = "Pitch an expansion", "revenue"
            value = h["expansion_win_rate"] * h["exp_median_by_size"].get(r.size_band, h["exp_median"])
            reason = (f"{r.segment_name} customer for {r.customer_days} days, "
                      f"${r.lifetime_revenue_usd:,.0f} so far, no open deal")
        rows.append({"object_type": "customer", "object_id": r.account_id, "account_id": r.account_id,
                     "account_name": r.account_name, "owner_rep_id": r.owner_rep_id,
                     "action": action, "category": category, "reason": reason, "value_usd": value})
    return rows


def main():
    con = duckdb.connect(str(DB_PATH))
    data_through = con.execute("SELECT data_through FROM raw.export_info").fetchone()[0]
    h = history(con)
    actions = pd.DataFrame(lead_actions(con, h) + deal_actions(con) + customer_actions(con, h))

    reps = con.execute("SELECT rep_id AS owner_rep_id, name AS owner_name, is_active FROM clean.sales_reps").df()
    actions = actions.merge(reps, on="owner_rep_id", how="left")
    # Customers still assigned to a rep who has left the company
    left = ~actions["is_active"]
    actions.loc[left, "reason"] += "; owner has left the company - reassign this account"
    actions = actions.rename(columns={"is_active": "owner_is_active"})

    actions["urgency"] = actions["action"].map(URGENCY).fillna(0.0)
    actions["priority"] = (actions["value_usd"] * actions["urgency"]).round(0)
    actions["value_usd"] = actions["value_usd"].round(0)
    actions = actions.sort_values(["category", "priority"], ascending=[False, False]).reset_index(drop=True)
    revenue = actions["category"] == "revenue"
    actions["rank_overall"] = actions["priority"].where(revenue).rank(ascending=False, method="first")
    actions["rank_for_rep"] = (actions[revenue].groupby("owner_rep_id")["priority"]
                               .rank(ascending=False, method="first"))
    actions.insert(0, "as_of_date", data_through)
    save_table(con, "analytics.next_best_actions", actions)

    print(f"Next best actions as of {data_through}: {len(actions):,} "
          f"({revenue.sum():,} revenue, {(~revenue).sum():,} hygiene)\n")
    summary = actions.groupby(["category", "action"]).agg(
        count=("action", "size"), value_usd=("value_usd", "sum")).sort_values(
        ["category", "value_usd"], ascending=[False, False])
    print(f"{'action':<40}{'count':>7}{'value at stake':>17}")
    for (category, action), r in summary.iterrows():
        value = f"${r.value_usd:,.0f}" if category == "revenue" else "-"
        print(f"{action:<40}{int(r['count']):>7}{value:>17}")
    orphaned = actions[~actions.owner_is_active]
    print(f"\n{len(orphaned)} actions belong to reps who have left "
          f"({(orphaned.object_type == 'customer').sum()} of them customers) - flagged for reassignment")

    print("\nTop 10 actions across the team:")
    for r in actions[revenue].head(10).itertuples():
        print(f"  {int(r.rank_overall):>2}. {r.owner_name:<20} {r.action:<38} {r.object_type:<8} "
              f"{r.account_name[:28]:<28} ${r.value_usd:>8,.0f}")

    busiest = actions[revenue & actions.owner_is_active].groupby("owner_name")["value_usd"].sum().idxmax()
    print(f"\nExample to-do list for {busiest} (top 5):")
    for r in actions[revenue & (actions.owner_name == busiest)].head(5).itertuples():
        print(f"  {int(r.rank_for_rep)}. {r.action} - {r.account_name}\n     why: {r.reason}")
    con.close()


if __name__ == "__main__":
    main()
