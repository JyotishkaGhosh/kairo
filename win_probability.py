"""
win_probability.py - Milestone 4a: how likely is each open deal to be won?

Learns from analytics.deal_daily_snapshots (one row per open deal per day,
only information known that day). Label: was the deal eventually won?

Why only "old enough" deals are used for training: a deal's outcome is only
certain once it has closed, and quick closes are more often wins. Training on
"deals that happened to be closed already" would therefore make the model too
optimistic about young deals. Instead we learn from deals created at least
DEAD_AFTER_DAYS ago: nearly all have closed, and any still open by then are
counted as lost (in the data, every win happened within ~6 months).

Steps:
  1. Backtest: build the model as it would have been BACKTEST_DAYS ago and
     test it on the next batch of deals, comparing it with the reps' own
     probability field.
  2. Retrain on everything known today, score every open deal, add simple
     risk flags (overdue close date, no buyer response, stuck in stage, ...).

Writes to data/kairo.duckdb:
  analytics.deal_scores     today's model win probability for every open deal
  analytics.model_metrics   backtest results (model = 'win_probability')

Usage (PowerShell):
  python win_probability.py
"""

from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load import DB_PATH, ROOT
from ml_utils import (MODEL_KINDS, compare_models, evaluate, fit, predict,
                      print_calibration, save_metrics, save_table)

DEAD_AFTER_DAYS = 200  # a deal still open this long after creation counts as lost
BACKTEST_DAYS = 90     # the backtest pretends the model was built this many days ago
TEST_COHORT_DAYS = 90  # backtest tests on deals created during this many days

CATEGORICAL = ["deal_type", "source", "product_tier", "industry", "size_band", "region", "stage"]
NUMERIC = ["stage_order", "amount_log", "amount_missing", "probability_pct", "age_log",
           "days_in_stage_log", "days_to_close", "is_close_date_past", "close_date_pushes",
           "activities_7d", "activities_30d", "buyer_responses_30d", "responses_total_log",
           "meetings_total", "days_since_activity", "days_since_response",
           "rep_win_rate", "rep_closed_log"]
FEATURES = CATEGORICAL + NUMERIC


def load_snapshots(con):
    """Deal snapshots + the owner's track record up to that day (point-in-time)."""
    df = con.execute("""
        WITH rep_daily AS (
            SELECT owner_rep_id, closed_date AS day, count(*) AS closed,
                   count(*) FILTER (WHERE is_won) AS won
            FROM clean.deals WHERE is_closed GROUP BY ALL
        ), rep_running AS (
            SELECT owner_rep_id, day,
                   sum(closed) OVER w AS rep_closed_deals, sum(won) OVER w AS rep_won_deals
            FROM rep_daily
            WINDOW w AS (PARTITION BY owner_rep_id ORDER BY day ROWS UNBOUNDED PRECEDING)
        )
        SELECT s.*, s.snapshot_date - CAST(s.days_since_created AS INTEGER) AS created_date,
               coalesce(r.rep_closed_deals, 0) AS rep_closed_deals,
               coalesce(r.rep_won_deals, 0) AS rep_won_deals
        FROM analytics.deal_daily_snapshots s
        ASOF LEFT JOIN rep_running r
            ON r.owner_rep_id = s.owner_rep_id AND r.day <= s.snapshot_date
    """).df()
    for col in ("snapshot_date", "created_date", "expected_close_date", "outcome_closed_date"):
        df[col] = pd.to_datetime(df[col])
    return add_features(df)


def add_features(df):
    """Turn snapshot columns into model inputs (same code for training and scoring)."""
    df = df.copy()
    age = df["days_since_created"]
    df["amount_missing"] = df["amount_usd"].isna().astype(int)
    df["amount_log"] = np.log1p(df["amount_usd"].fillna(0))
    df["age_log"] = np.log1p(age.clip(upper=400))
    df["days_in_stage_log"] = np.log1p(df["days_in_stage"].clip(upper=300))
    df["days_to_close"] = df["days_to_expected_close"].clip(-120, 240)
    df["is_close_date_past"] = df["is_close_date_past"].astype(int)
    df["close_date_pushes"] = df["close_date_pushes"].clip(upper=10)
    df["responses_total_log"] = np.log1p(df["buyer_responses_total"])
    # "never" = the whole life of the deal so far
    df["days_since_activity"] = df["days_since_last_activity"].fillna(age).clip(upper=120)
    df["days_since_response"] = df["days_since_buyer_response"].fillna(age).clip(upper=120)
    # Rep's win rate so far, pulled towards 30% while they have few closed deals
    df["rep_win_rate"] = (df["rep_won_deals"] + 0.3 * 10) / (df["rep_closed_deals"] + 10)
    df["rep_closed_log"] = np.log1p(df["rep_closed_deals"])
    return df


def labelled_as_of(snaps, day):
    """Training rows for a model built on `day`: deals old enough that the outcome is known."""
    day = pd.Timestamp(day)
    rows = snaps[(snaps["created_date"] <= day - pd.Timedelta(days=DEAD_AFTER_DAYS))
                 & (snaps["snapshot_date"] <= day)].copy()
    rows["label"] = (rows["outcome_is_won"] & (rows["outcome_closed_date"] <= day)).astype(int)
    return add_deal_weights(rows)


def add_deal_weights(rows):
    """Each deal appears once per open day. Its rows are near-copies, so weight
    them to count as one deal in total - otherwise the model thinks it has far
    more evidence than it really has and becomes overconfident."""
    rows["weight"] = 1.0 / rows.groupby("deal_id")["deal_id"].transform("size")
    return rows


def risk_flags(row, stuck_after):
    flags = []
    if row.is_close_date_past:
        flags.append("close date has passed")
    if row.close_date_pushes >= 2:
        flags.append(f"close date pushed {row.close_date_pushes}x")
    if row.days_since_response >= 21:
        flags.append(f"no buyer response in {int(row.days_since_response)}+ days")
    if row.days_in_stage > stuck_after.get(row.stage, 1e9):
        flags.append(f"stuck in {row.stage} for {row.days_in_stage} days")
    if row.rep_probability_pct - row.win_probability_pct >= 25:
        flags.append("rep far more optimistic than model")
    return "; ".join(flags)


def main():
    con = duckdb.connect(str(DB_PATH))
    data_through = con.execute("SELECT data_through FROM raw.export_info").fetchone()[0]
    snaps = load_snapshots(con)

    # ---------- 1. Backtest ----------
    cutoff = data_through - timedelta(days=BACKTEST_DAYS)
    first_test = pd.Timestamp(cutoff - timedelta(days=DEAD_AFTER_DAYS - 1))
    last_test = first_test + pd.Timedelta(days=TEST_COHORT_DAYS - 1)
    train = labelled_as_of(snaps, cutoff)
    test = labelled_as_of(snaps, data_through)
    test = test[(test["created_date"] >= first_test) & (test["created_date"] <= last_test)]
    print(f"Backtest: model built as of {cutoff}, trained on {train.deal_id.nunique():,} deals "
          f"created up to {(first_test - pd.Timedelta(days=1)).date()},")
    print(f"tested on the next {test.deal_id.nunique():,} deals (created {first_test.date()} .. "
          f"{last_test.date()}), every day they were open ({len(test):,} deal-days)\n")

    results, preds, chosen = compare_models(train, test, CATEGORICAL, NUMERIC, weight="weight")
    p = preds[chosen]
    rep_p = (test["probability_pct"] / 100).clip(0.01, 0.99)
    w = test["weight"]
    rep_results = evaluate(test["label"], rep_p, w)
    print(f"\n  For comparison, the reps' own probability field: AUC {rep_results['auc']:.3f}, "
          f"log loss {rep_results['log_loss']:.4f}")
    print(f"  Average predicted win chance: model {np.average(p, weights=w):.0%}, "
          f"reps {np.average(rep_p, weights=w):.0%}, actual {np.average(test['label'], weights=w):.0%}")
    print_calibration(test["label"], p, w, noun="deals")
    print("\n  The reps' probabilities, same test:")
    print_calibration(test["label"], rep_p.to_numpy(), w, noun="deals",
                      bins=(0, 0.15, 0.3, 0.5, 0.7, 1.0))

    # Evaluation only: the simulator's hidden true odds on the day each deal was created.
    truth = duckdb.sql(f"SELECT deal_id, true_win_prob FROM "
                       f"'{(ROOT / 'data' / 'sim_state' / 'deals_truth.parquet').as_posix()}'").df()
    day0 = test.assign(p=p)[test["days_since_created"] == 0].merge(truth, on="deal_id")
    print(f"\n  On the day deals were created: model AUC {roc_auc_score(day0.label, day0.p):.3f}, "
          f"simulator's hidden true odds AUC {roc_auc_score(day0.label, day0.true_win_prob):.3f}")

    # ---------- 2. Retrain on everything known today, score open deals ----------
    all_known = labelled_as_of(snaps, data_through)
    final = fit(MODEL_KINDS[chosen](CATEGORICAL, NUMERIC), all_known, FEATURES, "weight")
    # "Stuck" = longer in the stage than 90% of eventually-won deals ever were
    stuck_after = all_known[all_known["label"] == 1].groupby("stage")["days_in_stage"].quantile(0.9).to_dict()

    today = snaps[snaps["snapshot_date"] == pd.Timestamp(data_through)].copy()
    today["win_probability_pct"] = (predict(final, today, FEATURES) * 100).round(1)
    today["rep_probability_pct"] = today["probability_pct"]
    today["risk_flags"] = [risk_flags(r, stuck_after) for r in today.itertuples()]
    names = con.execute("""
        SELECT d.deal_id, d.deal_name, a.account_name, r.name AS owner_name
        FROM clean.deals d JOIN clean.accounts a USING (account_id)
        JOIN clean.sales_reps r ON r.rep_id = d.owner_rep_id""").df()
    scores = today.merge(names, on="deal_id")[[
        "deal_id", "deal_name", "account_name", "owner_rep_id", "owner_name", "deal_type",
        "stage", "amount_usd", "expected_close_date", "rep_probability_pct",
        "win_probability_pct", "risk_flags"]]
    scores["expected_close_date"] = scores["expected_close_date"].dt.date
    scores = scores.sort_values("win_probability_pct", ascending=False).reset_index(drop=True)
    scores.insert(0, "as_of_date", data_through)
    scores["model"] = chosen
    save_table(con, "analytics.deal_scores", scores)

    save_metrics(con, "win_probability", data_through, {
        "backtest_cutoff": str(cutoff), "test_deals": test.deal_id.nunique(),
        "chosen_model": chosen, "auc": results[chosen]["auc"],
        "log_loss": results[chosen]["log_loss"], "brier": results[chosen]["brier"],
        "rep_auc": rep_results["auc"], "rep_log_loss": rep_results["log_loss"],
        "rep_brier": rep_results["brier"]})

    amount = scores["amount_usd"].fillna(0)
    print(f"\nScored {len(scores)} open deals as of {data_through}. Pipeline ${amount.sum():,.0f}:")
    print(f"  weighted by rep probability   ${(amount * scores.rep_probability_pct / 100).sum():>12,.0f}")
    print(f"  weighted by model probability ${(amount * scores.win_probability_pct / 100).sum():>12,.0f}")
    print(f"  deals with at least one risk flag: {(scores.risk_flags != '').sum()}")
    print("\nBiggest open deals:")
    for r in scores.nlargest(5, "amount_usd").itertuples():
        print(f"  {r.deal_name[:38]:<38} {r.stage:<12} ${r.amount_usd:>9,.0f}  rep {r.rep_probability_pct:>3}%"
              f"  model {r.win_probability_pct:>4.1f}%  {r.risk_flags}")
    con.close()


if __name__ == "__main__":
    main()
