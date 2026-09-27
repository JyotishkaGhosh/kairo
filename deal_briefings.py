"""
deal_briefings.py - Milestone 6: short, LLM-written briefings for the most
important open deals.

For each of the top MAX_BRIEFINGS open deals (ranked by next_best_action.py),
we collect the facts the pipeline already knows - stage, amount, dates, model
vs rep probability, risk flags, recent activity, the recommended action - and
ask Claude to turn them into a briefing a sales rep can read in 20 seconds:
a headline, the situation, risks and next steps.

Claude only rewrites the facts it is given (it is told not to invent anything),
and returns structured JSON so the website can display it reliably.

Cost control: a briefing is only regenerated when something material about the
deal changed (stage, amount, close date, recommended action, risks, ...) or it
is older than REFRESH_AFTER_DAYS. Briefings are stored in
data/briefings/deal_briefings.json so they survive between daily runs.

API key: read from the ANTHROPIC_API_KEY environment variable - never put it in
code or in a file in this repo. In GitHub Actions it comes from a GitHub Secret.
Without a key the script keeps the existing briefings and generates none, so
the rest of the pipeline still works.

Writes:
  data/briefings/deal_briefings.json   the briefings (kept in git)
  analytics.deal_briefings             the same, in DuckDB

Usage (PowerShell):
  python deal_briefings.py             # generate / refresh briefings
  python deal_briefings.py --dry-run   # show what would be sent for the top deal; no API call
"""

import argparse
import hashlib
import json
import os
from datetime import date

import anthropic
import duckdb
import pandas as pd

from load import DB_PATH, ROOT
from ml_utils import save_table

MODEL = "claude-opus-5"
MAX_BRIEFINGS = 20        # briefings for the top-N deals by next-best-action priority
REFRESH_AFTER_DAYS = 7    # regenerate an unchanged deal's briefing after this many days
PRICE_PER_MTOK = {"input": 5.00, "output": 25.00}  # Claude Opus 5, USD per million tokens
BRIEFINGS_FILE = ROOT / "data" / "briefings" / "deal_briefings.json"

SYSTEM_PROMPT = """\
You write deal briefings for account executives at a B2B SaaS company. A rep
reads your briefing in about 20 seconds before a call or a pipeline review.

You receive the facts about one open deal as JSON, taken from the company's CRM
and its forecasting models. Write only from those facts:
- Never invent names, numbers, dates, quotes, competitors or events. If
  something isn't in the facts, don't mention it.
- Refer to absolute dates (e.g. "since 12 Aug") rather than "X days ago", so the
  briefing stays correct for a few days.
- "model_win_probability_pct" comes from a machine-learning model trained on
  past deals; "rep_probability_pct" is the rep's own estimate. When they differ
  a lot, say so plainly and neutrally.
- The recommended next action was chosen by the company's playbook; build the
  next steps around it, made concrete for this deal.
- Plain, direct language. No hype, no filler, no emojis.

Fields:
- headline: one line, at most 12 words, the single most important thing.
- situation: 2-3 sentences on where the deal stands and why.
- risks: 0-3 short items, most serious first (empty list if there are none).
- next_steps: 1-3 short, concrete actions, most important first."""

BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "situation": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "situation", "risks", "next_steps"],
    "additionalProperties": False,
}


def select_deals(con):
    return [r[0] for r in con.execute(f"""
        SELECT object_id FROM analytics.next_best_actions
        WHERE object_type = 'deal' AND category = 'revenue'
        ORDER BY priority DESC, object_id LIMIT {MAX_BRIEFINGS}""").fetchall()]


def deal_facts(con, deal_id):
    """Everything the briefing may use, as plain JSON-friendly values."""
    f = con.execute("""
        SELECT s.as_of_date, s.deal_id, s.deal_name, s.account_name, a.industry, a.size_band,
               a.region, a.is_customer, d.deal_type, d.source, d.product_tier, s.stage,
               s.amount_usd, s.expected_close_date, d.created_date, s.owner_name,
               s.rep_probability_pct, s.win_probability_pct AS model_win_probability_pct,
               round(f30.p_won * 100, 1) AS model_prob_won_within_30_days_pct, s.risk_flags,
               snap.close_date_pushes, snap.activities_30d, snap.buyer_responses_30d,
               snap.meetings_total, nba.action AS recommended_action,
               nba.reason AS recommended_action_reason,
               c.first_name || ' ' || c.last_name AS primary_contact, c.job_title AS contact_title,
               seg.segment_name AS customer_segment, seg.lifetime_revenue_usd AS customer_lifetime_revenue_usd
        FROM analytics.deal_scores s
        JOIN clean.deals d USING (deal_id)
        JOIN clean.accounts a ON a.account_id = d.account_id
        JOIN clean.contacts c ON c.contact_id = d.primary_contact_id
        JOIN analytics.deal_close_forecast f30 ON f30.deal_id = s.deal_id AND f30.horizon_days = 30
        JOIN analytics.deal_daily_snapshots snap
            ON snap.deal_id = s.deal_id AND snap.snapshot_date = s.as_of_date
        JOIN analytics.next_best_actions nba ON nba.object_type = 'deal' AND nba.object_id = s.deal_id
        LEFT JOIN analytics.customer_segments seg ON seg.account_id = d.account_id
        WHERE s.deal_id = ?""", [deal_id]).df().iloc[0].to_dict()

    changes = con.execute("""
        SELECT change_date, change_type, from_stage, to_stage, expected_close_date
        FROM clean.deal_stage_history
        WHERE deal_id = ? AND change_type IN ('created', 'stage_change', 'close_date_change')
        ORDER BY history_id""", [deal_id]).fetchall()
    f["history"] = [
        f"{d}: created in {to}" if kind == "created"
        else f"{d}: moved {frm} -> {to}" if kind == "stage_change"
        else f"{d}: expected close date changed to {close}"
        for d, kind, frm, to, close in changes]
    f["recent_activities"] = [
        f"{d}: {kind} ({direction}, {outcome})" for d, kind, direction, outcome in con.execute("""
            SELECT activity_date, activity_type, direction, outcome FROM clean.activities
            WHERE deal_id = ? ORDER BY activity_at DESC LIMIT 10""", [deal_id]).fetchall()]
    return {k: to_plain(v) for k, v in f.items() if not (v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)))}


def to_plain(v):
    if isinstance(v, (pd.Timestamp, date)):
        return pd.Timestamp(v).date().isoformat()
    if hasattr(v, "item"):  # numpy number -> Python number
        return v.item()
    return v


def material_key(facts):
    """Fingerprint of what matters. Unchanged fingerprint = the old briefing is still good."""
    keep = ["stage", "amount_usd", "expected_close_date", "owner_name", "recommended_action",
            "risk_flags", "meetings_total", "close_date_pushes"]
    material = {k: facts.get(k) for k in keep}
    material["win_prob_band"] = round(facts["model_win_probability_pct"] / 10)
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def user_message(facts):
    return ("Write the briefing for this deal. Facts (JSON):\n\n"
            + json.dumps(facts, indent=2, ensure_ascii=False))


def generate(client, facts):
    """One API call -> (briefing dict, usage) or (None, reason)."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message(facts)}],
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": BRIEFING_SCHEMA}},
        # If Claude's safety classifiers decline, re-run on Anthropic's recommended fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason != "end_turn":
        return None, f"stopped early ({response.stop_reason})"
    text = [b.text for b in response.content if b.type == "text"]
    if not text:
        return None, "no text in the response"
    return json.loads(text[-1]), response.usage


def load_existing():
    if BRIEFINGS_FILE.exists():
        return {b["deal_id"]: b for b in json.loads(BRIEFINGS_FILE.read_text(encoding="utf-8"))}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Write LLM deal briefings.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be sent for the top deal; no API call")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH))
    data_through = con.execute("SELECT data_through FROM raw.export_info").fetchone()[0]
    deal_ids = select_deals(con)
    existing = load_existing()

    # Decide per deal: reuse the stored briefing, or (re)generate it
    plan = []
    for deal_id in deal_ids:
        facts = deal_facts(con, deal_id)
        key = material_key(facts)
        old = existing.get(deal_id)
        fresh = (old is not None and old["material_key"] == key
                 and (data_through - date.fromisoformat(old["generated_on"])).days < REFRESH_AFTER_DAYS)
        plan.append((deal_id, facts, key, old if fresh else None))
    to_generate = [p for p in plan if p[3] is None]
    print(f"Briefings as of {data_through}: top {len(deal_ids)} deals, "
          f"{len(deal_ids) - len(to_generate)} still up to date, {len(to_generate)} to (re)write")

    if args.dry_run:
        deal_id, facts, key, _ = plan[0]
        print(f"\n--- DRY RUN: request for deal {deal_id} (model {MODEL}); nothing is sent ---\n")
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT + "\n")
        print("USER MESSAGE:\n" + user_message(facts))
        return

    client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
    if client is None and to_generate:
        print("ANTHROPIC_API_KEY is not set: keeping existing briefings, writing no new ones.")

    briefings, tokens_in, tokens_out, failed = [], 0, 0, 0
    for deal_id, facts, key, reuse in plan:
        if reuse is not None:
            briefings.append(reuse)
            continue
        if client is None:
            if deal_id in existing:
                briefings.append(existing[deal_id])  # stale, but better than nothing
            continue
        try:
            result, usage = generate(client, facts)
        except anthropic.RateLimitError:
            result, usage = None, "rate limited (after automatic retries)"
        except anthropic.APIStatusError as e:
            result, usage = None, f"API error {e.status_code}: {e.message}"
        except anthropic.APIConnectionError:
            result, usage = None, "could not reach the API"
        except json.JSONDecodeError:
            result, usage = None, "response was not valid JSON"
        if result is None:
            failed += 1
            print(f"  deal {deal_id}: skipped - {usage}")
            if deal_id in existing:
                briefings.append(existing[deal_id])
            continue
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens
        briefings.append({"deal_id": deal_id, "deal_name": facts["deal_name"],
                          "generated_on": data_through.isoformat(), "model": MODEL,
                          "material_key": key, **result,
                          "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens})
        print(f"  deal {deal_id}: {result['headline']}")

    BRIEFINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BRIEFINGS_FILE.write_text(json.dumps(briefings, indent=2, ensure_ascii=False), encoding="utf-8")
    table = pd.DataFrame(briefings, columns=[
        "deal_id", "deal_name", "generated_on", "model", "material_key", "headline",
        "situation", "risks", "next_steps", "input_tokens", "output_tokens"])
    for col in ("risks", "next_steps"):  # lists -> JSON text for DuckDB
        table[col] = table[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else None)
    save_table(con, "analytics.deal_briefings", table)
    con.close()

    cost = tokens_in / 1e6 * PRICE_PER_MTOK["input"] + tokens_out / 1e6 * PRICE_PER_MTOK["output"]
    written = len(to_generate) - failed if client else 0
    print(f"\nSaved {len(briefings)} briefings to {BRIEFINGS_FILE} "
          f"({written} new, {failed} failed). Tokens: {tokens_in:,} in / {tokens_out:,} out, "
          f"about ${cost:.2f}")


if __name__ == "__main__":
    main()
