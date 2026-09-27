"""
revenue_forecast.py - Milestone 4b: how much revenue will we win in the next
30 and 90 days - and how accurate has that forecast been in the past?

Forecast for the window (today, today + H days] =
    existing pipeline:  for each open deal, P(won within H days) x expected amount
  + new pipeline:       revenue from deals that don't exist yet but will be
                        created and won inside the window: past revenue per
                        new lead x the number of leads currently coming in

P(won within H days) comes from a model trained on deal snapshots. Its label
("was this deal won within H days of that day?") is fully known for every
snapshot at least H days old, so no deal has to be left out or guessed.

Backtest: every BACKTEST_STEP_DAYS days in the past, the whole forecast is
rebuilt using ONLY data known that day, then compared with what really closed.
Two classic methods are scored the same way for comparison:
  - rep-weighted pipeline: amount x rep probability, for deals the reps
    expect to close inside the window (what most CRMs show)
  - run-rate: "the next H days will look like the last H days"

Writes to data/kairo.duckdb:
  analytics.revenue_forecast      today's forecast per horizon, with a range
  analytics.forecast_backtest     every past forecast vs what really happened
  analytics.deal_close_forecast   per open deal: P(won within H days), expected $
  analytics.model_metrics         backtest accuracy (model = 'revenue_forecast')

Usage (PowerShell):
  python revenue_forecast.py
"""

from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd

from load import DB_PATH
from ml_utils import MODEL_KINDS, fit, predict, save_metrics, save_table
from win_probability import CATEGORICAL, FEATURES, NUMERIC, load_snapshots

HORIZONS = [30, 90]
BACKTEST_STEP_DAYS = 14
# An H-day model can only learn from snapshots at least H days old. Only count
# backtest forecasts whose model had at least this many days of history.
MIN_TRAINING_DAYS = 180
NEW_PIPELINE_LOOKBACK = 180  # days of history used for "revenue per new lead"
NEW_PIPELINE_SKIP = 120      # ignore the first months of the CRM (pipeline was still filling up)


def horizon_rows(snaps, day, horizon):
    """Training rows for a model built on `day`: snapshots at least `horizon` days old."""
    day = pd.Timestamp(day)
    rows = snaps[snaps["snapshot_date"] <= day - pd.Timedelta(days=horizon)].copy()
    rows["label"] = (rows["outcome_is_won"] & (rows["outcome_closed_date"]
                     <= rows["snapshot_date"] + pd.Timedelta(days=horizon))).astype(int)
    # No per-deal weights here (unlike win_probability.py): the forecast adds up
    # every deal open on a given day, so a long-open deal really should count
    # once per day. Backtested: per-deal weights made the 30-day forecast worse.
    return rows


def new_pipeline_estimate(deals, lead_dates, day, horizon, crm_start):
    """Revenue from deals not created yet = past revenue per new lead x current lead volume.

    Past revenue per lead: in earlier H-day windows (fully finished by `day`),
    revenue won inside the window from deals created inside it, divided by the
    leads that arrived in it. Scaling by the leads of the last H days makes the
    estimate follow growth and slow-downs in lead volume.
    """
    day = pd.Timestamp(day)
    won_total, leads_total = 0.0, 0
    start = day - pd.Timedelta(days=horizon)  # latest window that has fully ended by `day`
    earliest = max(day - pd.Timedelta(days=horizon + NEW_PIPELINE_LOOKBACK),
                   crm_start + pd.Timedelta(days=NEW_PIPELINE_SKIP))
    while start >= earliest:
        end = start + pd.Timedelta(days=horizon)
        won = deals[(deals.created_date > start) & deals.is_won
                    & (deals.closed_date > start) & (deals.closed_date <= end)]
        won_total += won.amount_usd.sum()
        leads_total += ((lead_dates > start) & (lead_dates <= end)).sum()
        start -= pd.Timedelta(days=7)
    if leads_total == 0:
        return 0.0
    recent_leads = ((lead_dates > day - pd.Timedelta(days=horizon)) & (lead_dates <= day)).sum()
    return float(won_total / leads_total * recent_leads)


def forecast_as_of(snaps, deals, lead_dates, day, horizon, kind, crm_start):
    """Everything needed for one forecast, using only information known on `day`."""
    day = pd.Timestamp(day)
    train = horizon_rows(snaps, day, horizon)
    model = fit(MODEL_KINDS[kind](CATEGORICAL, NUMERIC), train, FEATURES)

    # Expected amount if won: reps' amounts drift (discounts, optimism), so learn
    # the typical final/current ratio from deals already won, and use the tier's
    # typical won amount when the amount field is empty.
    won_deals = deals[deals.is_won & (deals.closed_date <= day)]
    pos = train[(train.label == 1) & train.amount_usd.notna()].merge(
        won_deals[["deal_id", "amount_usd"]].rename(columns={"amount_usd": "final_usd"}), on="deal_id")
    ratio = pos.final_usd.sum() / pos.amount_usd.sum()
    tier_median = won_deals.groupby("product_tier")["amount_usd"].median()

    open_now = snaps[snaps["snapshot_date"] == day].copy()
    open_now["p_won"] = predict(model, open_now, FEATURES)
    open_now["expected_amount"] = (open_now.amount_usd * ratio).fillna(
        open_now.product_tier.map(tier_median)).fillna(0)
    open_now["expected_usd"] = open_now.p_won * open_now.expected_amount

    window_end = day + pd.Timedelta(days=horizon)
    in_window = (open_now.expected_close_date > day) & (open_now.expected_close_date <= window_end)
    recent = deals[deals.is_won & (deals.closed_date > day - pd.Timedelta(days=horizon))
                   & (deals.closed_date <= day)]
    existing = open_now.expected_usd.sum()
    new = new_pipeline_estimate(deals, lead_dates, day, horizon, crm_start)
    summary = {
        "forecast_date": day.date(), "horizon_days": horizon, "window_end": window_end.date(),
        "existing_pipeline_usd": existing, "new_pipeline_usd": new, "forecast_usd": existing + new,
        "rep_weighted_usd": (open_now.amount_usd.fillna(0) * open_now.probability_pct / 100)[in_window].sum(),
        "run_rate_usd": recent.amount_usd.sum(),
    }
    return summary, open_now


def main():
    con = duckdb.connect(str(DB_PATH))
    data_through = pd.Timestamp(con.execute("SELECT data_through FROM raw.export_info").fetchone()[0])
    kind = con.execute("""SELECT json_extract_string(value, '$') FROM analytics.model_metrics
                          WHERE model = 'win_probability' AND metric = 'chosen_model'""").fetchone()
    kind = kind[0] if kind else "logistic regression"
    snaps = load_snapshots(con)
    deals = con.execute("""SELECT deal_id, product_tier, amount_usd, is_won,
                                  created_date, closed_date FROM clean.deals""").df()
    for col in ("created_date", "closed_date"):
        deals[col] = pd.to_datetime(deals[col])
    lead_dates = pd.to_datetime(con.execute("SELECT created_date FROM clean.leads").df()["created_date"])
    crm_start = snaps["snapshot_date"].min()
    print(f"Revenue forecast as of {data_through.date()} (model: {kind}, same as win probability)\n")

    # ---------- 1. Backtest ----------
    backtest = []
    for horizon in HORIZONS:
        day = crm_start + pd.Timedelta(days=MIN_TRAINING_DAYS + horizon)
        while day + pd.Timedelta(days=horizon) <= data_through:
            summary, _ = forecast_as_of(snaps, deals, lead_dates, day, horizon, kind, crm_start)
            won = deals[deals.is_won & (deals.closed_date > day)
                        & (deals.closed_date <= day + pd.Timedelta(days=horizon))]
            summary["actual_usd"] = won.amount_usd.sum()
            backtest.append(summary)
            day += pd.Timedelta(days=BACKTEST_STEP_DAYS)
    bt = pd.DataFrame(backtest)
    save_table(con, "analytics.forecast_backtest", bt)

    methods = {"forecast_usd": "Kairo forecast", "rep_weighted_usd": "rep-weighted pipeline",
               "run_rate_usd": "run-rate (last period)"}
    metrics, ranges = {}, {}
    for horizon in HORIZONS:
        h = bt[bt.horizon_days == horizon]
        print(f"Backtest, next {horizon} days: {len(h)} forecasts made between "
              f"{h.forecast_date.min()} and {h.forecast_date.max()}")
        print(f"  {'method':<26}{'avg error':>11}{'bias':>8}   (error = |forecast - actual| / actual)")
        for col, name in methods.items():
            err = (h[col] - h.actual_usd) / h.actual_usd
            print(f"  {name:<26}{err.abs().mean():>10.0%}{err.mean():>+8.0%}")
            metrics[f"mape_{horizon}d_{col.removesuffix('_usd')}"] = err.abs().mean()
            metrics[f"bias_{horizon}d_{col.removesuffix('_usd')}"] = err.mean()
        # Range for today's forecast: the error size that 80% of past forecasts stayed within
        ranges[horizon] = ((h.actual_usd - h.forecast_usd).abs() / h.forecast_usd).quantile(0.8)
        metrics[f"range_{horizon}d"] = ranges[horizon]
        print(f"  in 80% of past forecasts, actual revenue was within "
              f"+/-{ranges[horizon]:.0%} of the Kairo forecast\n")

    # ---------- 2. Today's forecast ----------
    rows, per_deal = [], []
    for horizon in HORIZONS:
        summary, open_now = forecast_as_of(snaps, deals, lead_dates, data_through, horizon, kind, crm_start)
        summary["low_usd"] = summary["forecast_usd"] * (1 - ranges[horizon])
        summary["high_usd"] = summary["forecast_usd"] * (1 + ranges[horizon])
        rows.append(summary)
        per_deal.append(open_now[["deal_id", "p_won", "expected_amount", "expected_usd"]]
                        .assign(horizon_days=horizon))
    fc = pd.DataFrame(rows)
    fc.insert(0, "as_of_date", data_through.date())
    save_table(con, "analytics.revenue_forecast", fc.drop(columns="forecast_date"))
    deal_fc = pd.concat(per_deal)
    deal_fc.insert(0, "as_of_date", data_through.date())
    save_table(con, "analytics.deal_close_forecast", deal_fc)
    save_metrics(con, "revenue_forecast", data_through.date(), {"model": kind, **metrics})

    print("Forecast of revenue won (new bookings):")
    for r in fc.itertuples():
        print(f"  next {r.horizon_days} days (to {r.window_end}): ${r.forecast_usd:>11,.0f}   "
              f"likely range ${r.low_usd:,.0f} - ${r.high_usd:,.0f}")
        print(f"      = ${r.existing_pipeline_usd:,.0f} from today's open deals "
              f"+ ${r.new_pipeline_usd:,.0f} from deals not created yet")
        print(f"      (rep-weighted pipeline says ${r.rep_weighted_usd:,.0f}; "
              f"last {r.horizon_days} days actually brought ${r.run_rate_usd:,.0f})")
    con.close()


if __name__ == "__main__":
    main()
