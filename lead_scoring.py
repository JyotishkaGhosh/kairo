"""
lead_scoring.py - Milestone 3: which open leads are most likely to become deals?

Learns from analytics.lead_daily_snapshots (built by transform.py). Each row
is one open lead on one day, described only with what was known that day;
the label is whether the lead was eventually converted into a deal.

Steps:
  1. Backtest: pretend it is BACKTEST_DAYS ago, train only on leads whose
     outcome was known by then, and test on the leads that arrived after.
     This shows how well the model would really have worked.
  2. Compare a simple, explainable model (logistic regression) with a more
     complex one (gradient boosting). Keep the simple one unless the complex
     one is clearly better.
  3. Retrain on everything known today and score every open lead (0-100),
     with a grade and the top reasons behind each score.

Writes to data/kairo.duckdb:
  analytics.lead_scores          today's score for every open lead
  analytics.lead_score_drivers   what pushes scores up or down (logistic model)
  analytics.model_metrics        backtest results (model = 'lead_scoring')

Usage (PowerShell):
  python lead_scoring.py
"""

from datetime import timedelta

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from load import DB_PATH, ROOT
from ml_utils import (MODEL_KINDS, compare_models, fit, logistic_model, plural, predict,
                      print_calibration, save_metrics, save_table)

BACKTEST_DAYS = 120   # the backtest pretends the model was built this many days ago
# A lead still open this many days after it arrived counts as "not converted".
# Without this rule, forgotten leads (never closed by the rep) would never get
# a label, the model would never learn from them, and it would score them
# like fresh leads.
DEAD_AFTER_DAYS = 60
GRADES = [(40, "A"), (25, "B"), (12, "C"), (0, "D")]  # score >= threshold -> grade

CATEGORICAL = ["source", "industry", "size_band", "region", "seniority"]
NUMERIC = ["is_existing_customer", "job_title_missing", "age_log", "touches_log",
           "days_since_touch", "responses_log", "responses_7d_log",
           "days_since_response", "meetings_log"]
FEATURES = CATEGORICAL + NUMERIC

# Features are grouped into "reasons" that a salesperson can understand.
REASON_GROUPS = {
    "source": ["source"], "industry": ["industry"], "size_band": ["size_band"],
    "region": ["region"], "seniority": ["seniority"],
    "customer": ["is_existing_customer"], "job_title": ["job_title_missing"],
    "age": ["age_log"], "rep_effort": ["touches_log", "days_since_touch"],
    "engagement": ["responses_log", "responses_7d_log", "days_since_response", "meetings_log"],
}


def reason_text(group, row):
    if group == "source":
        return f"{row.source} lead"
    if group == "industry":
        return f"{row.industry} industry"
    if group == "size_band":
        return f"{row.size_band} company"
    if group == "region":
        return f"{row.region} region"
    if group == "seniority":
        return f"{row.seniority} contact"
    if group == "customer":
        return "existing customer" if row.is_existing_customer else "not yet a customer"
    if group == "job_title":
        return "contact's job title unknown" if row.job_title_missing else "contact's job title known"
    if group == "age":
        return f"lead is {plural(row.days_since_created, 'day')} old"
    if group == "rep_effort":
        return plural(row.touches_total, 'rep touch', 'rep touches')
    if row.buyer_responses_total == 0:
        return "no buyer response yet"
    return (f"{plural(row.buyer_responses_total, 'buyer response')}, "
            f"last {plural(row.days_since_buyer_response, 'day')} ago")


def add_features(df):
    """Turn raw snapshot columns into model inputs (same code for training and scoring)."""
    df = df.copy()
    df["is_existing_customer"] = df["is_existing_customer"].astype(int)
    df["job_title_missing"] = df["job_title_missing"].astype(int)
    df["age_log"] = np.log1p(df["days_since_created"].clip(upper=180))
    df["touches_log"] = np.log1p(df["touches_total"])
    df["days_since_touch"] = df["days_since_last_touch"].fillna(df["days_since_created"]).clip(upper=90)
    df["responses_log"] = np.log1p(df["buyer_responses_total"])
    df["responses_7d_log"] = np.log1p(df["buyer_responses_7d"])
    df["days_since_response"] = df["days_since_buyer_response"].fillna(90).clip(upper=90)
    df["meetings_log"] = np.log1p(df["meetings_total"])
    return df


def labelled_as_of(snaps, day):
    """Rows whose final answer was already known on `day`, with that answer as label.

    Known = the lead was converted/disqualified by `day`, or it had been open
    for DEAD_AFTER_DAYS by then (counts as not converted). Only rows dated up
    to `day` are used, so nothing from after `day` leaks in.
    """
    day = pd.Timestamp(day)
    resolved = snaps["outcome_is_resolved"] & (snaps["outcome_resolved_date"] <= day)
    presumed_dead = snaps["created_date"] <= day - pd.Timedelta(days=DEAD_AFTER_DAYS)
    rows = snaps[(resolved | presumed_dead) & (snaps["snapshot_date"] <= day)].copy()
    rows["label"] = (rows["outcome_converted"] & (rows["outcome_resolved_date"] <= day)).astype(int)
    # Each lead appears once per open day; weight rows so every lead counts equally.
    rows["weight"] = 1.0 / rows.groupby("lead_id")["lead_id"].transform("size")
    return rows


def top20_conversion(test, p):
    """Of brand-new leads, how often do the top-scored 20% convert?"""
    new = test.assign(p=p)[test["days_since_created"] == 0]
    top = new[new["p"] >= new["p"].quantile(0.8)]
    return top["label"].mean(), new["label"].mean()


def explain(model, train, df):
    """Per lead: how much each reason group pushes the score up or down (log-odds)."""
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    names = prep.get_feature_names_out()
    center = prep.transform(train[FEATURES]).mean(axis=0)
    contrib = (prep.transform(df[FEATURES]) - center) * lr.coef_[0]
    groups = {}
    for group, features in REASON_GROUPS.items():
        cols = [i for i, n in enumerate(names)
                if any(n == f"num__{f}" or n.startswith(f"cat__{f}_") for f in features)]
        groups[group] = contrib[:, cols].sum(axis=1)
    return pd.DataFrame(groups, index=df.index)


def main():
    con = duckdb.connect(str(DB_PATH))
    data_through = con.execute("SELECT data_through FROM raw.export_info").fetchone()[0]
    snaps = add_features(con.execute("""
        SELECT *, snapshot_date - CAST(days_since_created AS INTEGER) AS created_date
        FROM analytics.lead_daily_snapshots""").df())
    for col in ("snapshot_date", "created_date", "outcome_resolved_date"):
        snaps[col] = pd.to_datetime(snaps[col])

    # ---------- 1. Backtest ----------
    cutoff = data_through - timedelta(days=BACKTEST_DAYS)
    last_test_day = data_through - timedelta(days=DEAD_AFTER_DAYS)
    train = labelled_as_of(snaps, cutoff)
    test = labelled_as_of(snaps, data_through)
    test = test[(test["created_date"] > pd.Timestamp(cutoff))
                & (test["created_date"] <= pd.Timestamp(last_test_day))]
    print(f"Backtest: model built as of {cutoff}, trained on {train.lead_id.nunique():,} leads "
          f"with a known outcome,")
    print(f"tested on the {test.lead_id.nunique():,} leads created "
          f"{cutoff + timedelta(days=1)} .. {last_test_day}\n")

    y, w = test["label"], test["weight"]
    results, preds, chosen = compare_models(train, test, CATEGORICAL, NUMERIC, weight="weight")
    p = preds[chosen]

    new_leads = test["days_since_created"] == 0
    auc_at_creation = roc_auc_score(y[new_leads], p[new_leads])
    top_rate, avg_rate = top20_conversion(test, p)
    print(f"  AUC on the day a lead arrives (no engagement yet): {auc_at_creation:.3f}")
    print(f"  Top-scored 20% of new leads converted {top_rate:.0%} vs {avg_rate:.0%} on average "
          f"({top_rate / avg_rate:.1f}x)")

    print_calibration(y, p, w, noun="leads")

    # Evaluation only: compare with the simulator's hidden true odds (never used for training).
    truth = duckdb.sql(f"SELECT lead_id, true_convert_prob FROM "
                       f"'{(ROOT / 'data' / 'sim_state' / 'leads_truth.parquet').as_posix()}'").df()
    t_new = test[new_leads].merge(truth, on="lead_id")
    oracle_auc = roc_auc_score(t_new["label"], t_new["true_convert_prob"])
    print(f"\n  For reference, the simulator's hidden true odds reach AUC {oracle_auc:.3f} on arrival day")
    print("  (the best any model could do with only the information available on arrival).")

    # ---------- 2. Retrain on everything known today, score open leads ----------
    all_known = labelled_as_of(snaps, data_through)
    final = fit(MODEL_KINDS[chosen](CATEGORICAL, NUMERIC), all_known, FEATURES, "weight")
    explainer = (final if chosen == "logistic regression"
                 else fit(logistic_model(CATEGORICAL, NUMERIC), all_known, FEATURES, "weight"))

    today = snaps[snaps["snapshot_date"] == pd.Timestamp(data_through)].merge(
        con.execute("SELECT lead_id, owner_rep_id FROM clean.leads").df(), on="lead_id")
    today["score"] = (predict(final, today, FEATURES) * 100).round(1)
    today["grade"] = today["score"].map(lambda s: next(g for t, g in GRADES if s >= t))
    contrib = explain(explainer, all_known, today)
    today["top_reasons"] = [
        "; ".join(("+ " if contrib.at[i, g] > 0 else "- ") + reason_text(g, row)
                  for g in contrib.loc[i].abs().nlargest(3).index)
        for i, row in today.iterrows()]

    scores = today[["lead_id", "account_id", "owner_rep_id", "source", "industry", "size_band",
                    "days_since_created", "buyer_responses_total", "score", "grade", "top_reasons"]]
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    scores.insert(0, "as_of_date", data_through)
    scores["model"] = chosen
    save_table(con, "analytics.lead_scores", scores)

    # What drives the score (logistic regression coefficients, in log-odds per unit / per category).
    lr = explainer.named_steps["model"]
    drivers = pd.DataFrame({"feature": explainer.named_steps["prep"].get_feature_names_out(),
                            "effect": lr.coef_[0]})
    drivers["feature"] = drivers["feature"].str.replace(r"^(cat|num)__", "", regex=True)
    drivers = drivers.sort_values("effect", ascending=False)
    save_table(con, "analytics.lead_score_drivers", drivers)

    save_metrics(con, "lead_scoring", data_through, {
        "backtest_cutoff": str(cutoff), "test_leads": test.lead_id.nunique(),
        "chosen_model": chosen, "auc": results[chosen]["auc"],
        "auc_at_creation": auc_at_creation, "log_loss": results[chosen]["log_loss"],
        "baseline_log_loss": results["baseline (everyone gets the average)"]["log_loss"],
        "brier": results[chosen]["brier"], "top20_conversion_rate": top_rate,
        "avg_conversion_rate": avg_rate, "oracle_auc_at_creation": oracle_auc})

    print(f"\nScored {len(scores):,} open leads as of {data_through}:")
    print("  " + ", ".join(f"{g}: {n}" for g, n in scores["grade"].value_counts().sort_index().items()))
    print("\nTop 5 leads right now:")
    for r in scores.head(5).itertuples():
        print(f"  lead {r.lead_id:>5}  score {r.score:>5.1f} ({r.grade})  {r.top_reasons}")
    print("\nStrongest drivers (log-odds; + raises the score, - lowers it):")
    for r in pd.concat([drivers.head(5), drivers.tail(5)]).itertuples():
        print(f"  {r.effect:+.2f}  {r.feature}")
    con.close()


if __name__ == "__main__":
    main()
