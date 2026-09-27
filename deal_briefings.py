"""
deal_briefings.py - Milestone 6: short, LLM-written briefings for the most
important open deals, using Google Gemini (free tier).

For each of the top MAX_BRIEFINGS open deals (ranked by next_best_action.py),
Gemini writes three short sentences a rep can read in seconds:
  situation  stage, amount and the model's win probability
  why        the model's top 2 reasons (from win_probability.py's per-deal
             explanations) plus up to two risk signals: days since last
             activity / buyer response, close-date slips, or a rep forecast
             far above the model
  action     the next best action (from next_best_action.py), made specific

Gemini only gets those fields, is told to use only the numbers given and never
invent anything, and returns structured JSON so the website can show it reliably.

Which model: at the start of each run we ask Google which models this API key
can use and pick the newest stable Flash-Lite model (usually less busy than
Flash), plus one fallback model. GEMINI_MODEL / GEMINI_FALLBACK_MODEL override
the choice. If the list can't be fetched, DEFAULT_MODELS are used.

Busy / rate-limited Gemini (HTTP 503 "high demand", 429 "too many requests"):
  - each model is retried after RETRY_WAITS seconds;
  - if it still fails, the fallback model is tried for that deal, and the busy
    model is skipped for the rest of this run;
  - the whole step stops after TIME_BUDGET_SECONDS;
  - any deal that couldn't be briefed keeps its previous briefing (the website
    shows "briefing unavailable" if there is none). This step never makes the
    daily pipeline fail.
Note: on the free tier Google may use what you send to improve its products -
fine here, because all CRM data is simulated.

Quota control: a briefing is only regenerated when something material about
the deal changed (stage, amount, close date, recommended action, risks, ...)
or it is older than REFRESH_AFTER_DAYS. Briefings are stored in
data/briefings/deal_briefings.json so they survive between daily runs.

Settings (environment variables - never put the key in code or in a file):
  GEMINI_API_KEY               your Google AI Studio API key (a GitHub Secret in Actions)
  GEMINI_MODEL                 optional: main model (default: chosen automatically)
  GEMINI_FALLBACK_MODEL        optional: fallback model (default: chosen automatically)
  GEMINI_TIME_BUDGET_SECONDS   optional: time limit for the whole step (default 300)

Writes:
  data/briefings/deal_briefings.json   the briefings (kept in git)
  analytics.deal_briefings             the same, in DuckDB

Usage (PowerShell):
  python deal_briefings.py                 # generate / refresh briefings
  python deal_briefings.py --list-models   # show the models your key can use + which are chosen
  python deal_briefings.py --dry-run       # show what would be sent for the top deal; no API call
  python deal_briefings.py --fake          # test run with a built-in stand-in for Gemini (no key
                                           # needed); writes deal_briefings.fake.json only
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import date
from types import SimpleNamespace

import duckdb
import pandas as pd
from google import genai
from google.genai import errors, types

from load import DB_PATH, ROOT
from ml_utils import save_table

# Used only if the list of models can't be fetched (main first, then fallback)
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
# Model names containing any of these are never picked automatically
SKIP_MODEL_WORDS = ("preview", "exp", "tts", "image", "live", "audio", "embedding", "thinking")
MAX_BRIEFINGS = 20        # briefings for the top-N deals by next-best-action priority
REFRESH_AFTER_DAYS = 7    # regenerate an unchanged deal's briefing after this many days
PAUSE_SECONDS = 5         # wait between requests, to stay under the free tier's per-minute limit
RETRY_WAITS = [5, 15, 30]  # seconds to wait before each retry of the same model
RETRYABLE_CODES = {429, 500, 502, 503, 504}  # too many requests / temporarily unavailable
REQUEST_TIMEOUT_MS = 60_000  # one request may take at most 60 s
TIME_BUDGET_SECONDS = int(os.environ.get("GEMINI_TIME_BUDGET_SECONDS") or 300)
BRIEFINGS_DIR = ROOT / "data" / "briefings"

PROMPT_VERSION = 5 # bump when the prompt or output format changes: every briefing is rewritten
REP_OPTIMISM_GAP = 25  # rep forecast this many points above the model = a risk (same as win_probability.py)

SYSTEM_PROMPT = """\
You write short deal briefings for sales reps at a B2B software company.

You get the facts about one open deal as JSON. They come from the CRM, a
machine-learning model that estimates the deal's chance of being won, and a
playbook that picks the next best action. Write exactly ONE sentence for each
field - three sentences in total, never more:

situation: the deal's stage, its amount and the model's win probability
  (from "situation").
why: ONE sentence with the two reasons in "why.top_reasons". For each reason,
  give its "detail" (with its numbers) and say whether it raises or lowers the
  win probability. Then add up to two risk signals from "why.risk_signals"
  (the most important first), if there are any: many days since the last
  activity or buyer response ("never" means it has not happened), a close
  date that slipped (close_date_slipped_times of 2 or more) or has passed,
  the rep's forecast being much higher than the model's, or the deal being
  stuck in its stage. Join everything with commas, "and" or "but" - no second
  sentence. If there is no real risk signal, do not make one up.
action: the next step from "action.next_best_action", made specific with the
  numbers in "action.why_this_action". Do not repeat the rep's and the model's
  percentages here - they are shown next to the briefing separately.

Example of the style only (a different, made-up deal - never copy its facts):
{"situation": "This deal is in Proposal for $58,000, and the model's win probability is 31%.",
 "why": "Only 1 buyer response in the last 30 days lowers the win probability and the rep's forecast of 70% raises it, but the close date has slipped 3 times and the rep's forecast is 39 points above the model.",
 "action": "Re-qualify the deal before it slips again: the model gives it a 31% chance, well below the rep's 70%."}

Rules:
- Use only the facts and numbers given. Never invent names, numbers, dates,
  reasons or events.
- Keep numbers as given. Write percentages as whole numbers (e.g. 47%) and
  money as e.g. $226,400.
- Plain, simple English that a new sales rep understands. "why" under about
  45 words, the other two under about 30. No abbreviations such as "3x".
  No hype or filler words (such as "exciting", "huge", "crucial",
  "significant", "strong momentum"), no exclamation marks, no emojis.
- Call the model's number "the model's win probability" and the rep's number
  "the rep's forecast" (never "the rep's estimate")."""

BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "situation": {"type": "string"},
        "why": {"type": "string"},
        "action": {"type": "string"},
    },
    "required": ["situation", "why", "action"],
    "additionalProperties": False,
}


class ModelUnavailable(Exception):
    """A model kept answering 429/5xx after all retries."""


class OutOfTime(Exception):
    """The time budget for the whole step is used up."""


class KeyProblem(Exception):
    """The API key is missing, invalid or not allowed - no point trying further."""


# ---------------------------------------------------------------------------
# Deal facts (what Gemini is allowed to use)
# ---------------------------------------------------------------------------
def select_deals(con):
    return [r[0] for r in con.execute(f"""
        SELECT object_id FROM analytics.next_best_actions
        WHERE object_type = 'deal' AND category = 'revenue'
        ORDER BY priority DESC, object_id LIMIT {MAX_BRIEFINGS}""").fetchall()]


def deal_facts(con, deal_id):
    """The facts Gemini may use, grouped by the three sentences it writes."""
    r = con.execute("""
        SELECT s.as_of_date, s.deal_name, s.account_name, s.owner_name, s.stage, s.amount_usd,
               s.expected_close_date, s.win_probability_pct, s.rep_probability_pct,
               s.risk_flags, s.top_reasons,
               snap.days_since_last_activity, snap.days_since_buyer_response,
               snap.close_date_pushes, snap.is_close_date_past,
               nba.action, nba.reason AS action_reason
        FROM analytics.deal_scores s
        JOIN analytics.deal_daily_snapshots snap
            ON snap.deal_id = s.deal_id AND snap.snapshot_date = s.as_of_date
        JOIN analytics.next_best_actions nba ON nba.object_type = 'deal' AND nba.object_id = s.deal_id
        WHERE s.deal_id = ?""", [deal_id]).df().iloc[0]
    days = lambda v: "never" if pd.isna(v) else int(v)
    gap = int(round(r.rep_probability_pct - r.win_probability_pct))
    # The other risk flags repeat the fields below in short-hand ("pushed 3x"); only "stuck" is new
    stuck = next((f for f in (r.risk_flags or "").split("; ") if f.startswith("stuck in")), None)
    return {
        "as_of_date": to_plain(r.as_of_date),
        "deal_name": r.deal_name,
        "account_name": r.account_name,
        "owner_name": r.owner_name,
        "situation": {
            "stage": r.stage,
            "amount_usd": None if pd.isna(r.amount_usd) else int(r.amount_usd),
            "model_win_probability_pct": int(round(r.win_probability_pct)),
            "expected_close_date": to_plain(r.expected_close_date),
        },
        "why": {
            # The two reason groups that move this deal's win probability most,
            # compared with an average open deal (from win_probability.py)
            "top_reasons": [{"factor": x["factor"], "effect": f"{x['effect']} the win probability",
                             "detail": x["detail"]} for x in json.loads(r.top_reasons)[:2]],
            "risk_signals": {
                "days_since_last_activity": days(r.days_since_last_activity),
                "days_since_last_buyer_response": days(r.days_since_buyer_response),
                "close_date_slipped_times": int(r.close_date_pushes),
                "close_date_has_passed": bool(r.is_close_date_past),
                "rep_forecast_pct": int(r.rep_probability_pct),
                "rep_forecast_minus_model_pct_points": gap,
                "rep_much_more_optimistic_than_model": gap >= REP_OPTIMISM_GAP,
                "stuck_in_stage": stuck,  # e.g. "stuck in Negotiation for 48 days", or null
            },
        },
        "action": {"next_best_action": r.action, "why_this_action": r.action_reason},
    }


def to_plain(v):
    if isinstance(v, (pd.Timestamp, date)):
        return pd.Timestamp(v).date().isoformat()
    if hasattr(v, "item"):  # numpy number -> Python number
        return v.item()
    return v


def material_key(facts):
    """Fingerprint of what matters. Unchanged fingerprint = the old briefing is still good.

    Day counts are put into bands (this week / 1-2 weeks / 2-4 weeks / longer),
    so a briefing isn't rewritten every day just because one more day passed.
    """
    s, why = facts["situation"], facts["why"]
    band = lambda d: -1 if d == "never" else 0 if d < 7 else 1 if d < 14 else 2 if d < 30 else 3
    risk = why["risk_signals"]
    material = {
        "prompt_version": PROMPT_VERSION,
        "stage": s["stage"], "amount": s["amount_usd"], "close": s["expected_close_date"],
        "win_prob_band": round(s["model_win_probability_pct"] / 10),
        "owner": facts["owner_name"], "action": facts["action"]["next_best_action"],
        "reasons": [(x["factor"], x["effect"]) for x in why["top_reasons"]],
        "activity_band": band(risk["days_since_last_activity"]),
        "response_band": band(risk["days_since_last_buyer_response"]),
        "slips": risk["close_date_slipped_times"], "passed": risk["close_date_has_passed"],
        "optimistic": risk["rep_much_more_optimistic_than_model"],
        "stuck": risk["stuck_in_stage"] is not None,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def user_message(facts):
    return ("Write the briefing for this deal. Facts (JSON):\n\n"
            + json.dumps(facts, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Choosing models
# ---------------------------------------------------------------------------
def list_models(client):
    """Names of the models this key can use to generate text."""
    return sorted(m.name.removeprefix("models/") for m in client.models.list()
                  if "generateContent" in (m.supported_actions or []))


def version_key(name):
    """'gemini-3.5-flash-lite' -> (3, 5): newer versions sort higher."""
    match = re.match(r"gemini-(\d+(?:\.\d+)*)", name)
    return tuple(int(p) for p in match.group(1).split(".")) if match else ()


def choose_models(available):
    """(main, fallback, how it was chosen). Env variables win over the automatic choice."""
    usable = [n for n in available
              if n.startswith("gemini-") and "flash" in n and not any(w in n for w in SKIP_MODEL_WORDS)]
    rank = lambda n: (version_key(n), -len(n))  # newest first; plain alias before "-001" style names
    lite = sorted((n for n in usable if "flash-lite" in n), key=rank, reverse=True)
    flash = sorted((n for n in usable if "flash-lite" not in n), key=rank, reverse=True)
    automatic = lite + flash  # Flash-Lite first (less busy), then regular Flash
    if not automatic:
        automatic = DEFAULT_MODELS
    main = os.environ.get("GEMINI_MODEL") or automatic[0]
    fallback = os.environ.get("GEMINI_FALLBACK_MODEL") or next((n for n in automatic if n != main), None)
    how = ("GEMINI_MODEL" if os.environ.get("GEMINI_MODEL") else
           f"chosen automatically from {len(available)} models your key can use" if usable else
           "built-in default (no suitable model in your key's list)" if available else
           "built-in default (model list not available)")
    return main, fallback, how


# ---------------------------------------------------------------------------
# Calling Gemini
# ---------------------------------------------------------------------------
class Clock:
    """Real time. The fake mode swaps in a virtual clock so tests don't really wait."""
    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def ask_model(client, model, facts, clock, deadline):
    """One briefing from one model, with retries.

    Returns (briefing, token counts) or (None, reason) for a non-retryable
    problem. Raises ModelUnavailable if it stays busy, OutOfTime at the limit.
    """
    for attempt in range(len(RETRY_WAITS) + 1):
        if clock.now() >= deadline:
            raise OutOfTime
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message(facts),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=BRIEFING_SCHEMA,
                ),
            )
            break
        except errors.APIError as e:
            if e.code in (401, 403) or "API key" in (e.message or ""):
                raise KeyProblem(f"{e.code}: {e.message}") from e
            if e.code == 404:  # model name doesn't exist (e.g. a typo in GEMINI_MODEL)
                raise ModelUnavailable("404 model not found") from e
            if e.code not in RETRYABLE_CODES:
                return None, f"error {e.code}: {e.message}"
            problem = f"{e.code} ({'too many requests' if e.code == 429 else 'busy / unavailable'})"
        except Exception as e:  # timeouts, dropped connections, ...
            problem = type(e).__name__
        if attempt == len(RETRY_WAITS):
            raise ModelUnavailable(problem)
        wait = RETRY_WAITS[attempt]
        if clock.now() + wait >= deadline:
            raise OutOfTime
        print(f"    {model}: {problem}; waiting {wait}s, then retry {attempt + 1}/{len(RETRY_WAITS)}")
        clock.sleep(wait)

    finish = response.candidates[0].finish_reason if response.candidates else None
    if finish != types.FinishReason.STOP:
        return None, f"stopped early ({finish.name if finish else 'no answer'})"
    if not response.text:
        return None, "empty answer"
    try:
        briefing = json.loads(response.text)
    except json.JSONDecodeError:
        return None, "answer was not valid JSON"
    usage = response.usage_metadata
    tokens = {"input_tokens": usage.prompt_token_count or 0,
              "output_tokens": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)}
    return briefing, tokens


def rep_vs_model(facts):
    """Shown next to the briefing (not written by Gemini): the rep's and the model's numbers."""
    return (f"rep says {facts['why']['risk_signals']['rep_forecast_pct']}%, "
            f"model says {facts['situation']['model_win_probability_pct']}%")


def current(briefing, facts):
    """An existing briefing with today's rep-vs-model numbers and no leftover stale mark."""
    b = {k: v for k, v in briefing.items() if k not in ("stale", "stale_reason")}
    return {**b, "rep_vs_model": rep_vs_model(facts)}


def write_briefings(client, models, plan, existing, data_through, clock, label):
    """Try to (re)write every stale briefing within the time budget. Never raises."""
    deadline = clock.now() + TIME_BUDGET_SECONDS
    benched = set()  # models that stayed busy this run: don't wait on them again
    stop_reason = None
    briefings, stats = [], {"written": 0, "failed": 0, "tokens_in": 0, "tokens_out": 0, "by_model": {}}
    requests_made = 0

    for deal_id, facts, key, reuse in plan:
        if reuse is not None:
            briefings.append(current(reuse, facts))
            continue
        result, failure = None, None
        if client is not None and stop_reason is None:
            try:
                for model in [m for m in models if m and m not in benched]:
                    if requests_made:
                        if clock.now() + PAUSE_SECONDS >= deadline:
                            raise OutOfTime
                        clock.sleep(PAUSE_SECONDS)  # stay under the per-minute limit
                    requests_made += 1
                    try:
                        result, info = ask_model(client, model, facts, clock, deadline)
                    except ModelUnavailable as e:
                        benched.add(model)
                        others = [m for m in models if m and m not in benched]
                        print(f"  {model} not usable in this run ({e})"
                              + (f"; using {others[0]} from now on" if others else ""))
                        continue
                    if result is not None:
                        used = model
                        break
                    print(f"  deal {deal_id}: {model} gave no usable briefing ({info})")
                    failure = f"{model} gave no usable briefing ({info})"
                if result is None and not [m for m in models if m and m not in benched]:
                    stop_reason = "every model is busy or out of quota"
            except OutOfTime:
                stop_reason = f"the {TIME_BUDGET_SECONDS // 60}-minute time limit was reached"
            except KeyProblem as e:
                stop_reason = f"Gemini rejected the API key ({e}) - check the GEMINI_API_KEY secret"
            except Exception as e:  # anything unexpected: keep going with old briefings
                stop_reason = f"unexpected error ({type(e).__name__}: {e})"
            if stop_reason:
                print(f"  Stopping briefings: {stop_reason}. Remaining deals keep their previous briefing.")

        if result is None:
            if client is not None:
                stats["failed"] += 1
            if deal_id in existing:
                # Keep the previous briefing, but say it was written from older facts - and why
                reason = ("GEMINI_API_KEY is not set" if client is None
                          else stop_reason or failure or "every model is busy or out of quota")
                briefings.append({**current(existing[deal_id], facts), "stale": True,
                                  "stale_reason": f"not rewritten on {data_through.isoformat()}: {reason}"})
            continue
        stats["written"] += 1
        stats["tokens_in"] += info["input_tokens"]
        stats["tokens_out"] += info["output_tokens"]
        stats["by_model"][used] = stats["by_model"].get(used, 0) + 1
        briefings.append({"deal_id": deal_id, "deal_name": facts["deal_name"],
                          "generated_on": data_through.isoformat(), "model": label or used,
                          "material_key": key, **result, "rep_vs_model": rep_vs_model(facts), **info})
        print(f"  deal {deal_id} ({used}): {result['situation'][:100]}")
    return briefings, stats


# ---------------------------------------------------------------------------
# Fake Gemini for testing without a key
# ---------------------------------------------------------------------------
class FakeClock:
    """Virtual time: sleeping just moves the clock forward."""
    def __init__(self):
        self.t = 0.0
    def now(self):
        return self.t
    def sleep(self, seconds):
        self.t += seconds


class FakeGemini:
    """Stand-in for the Gemini client (same call and response shapes as the real SDK).

    Its "briefings" are filled-in templates built from exactly the facts Gemini
    receives - handy to check the inputs and the page, but not real AI writing.

    It offers a realistic list of models, and the first Flash-Lite model is
    always "experiencing high demand" (503) - so a fake run shows the retries,
    the switch to the fallback model and the time accounting.
    """
    BUSY_MODEL = "gemini-3.5-flash-lite"
    MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
              "gemini-3-flash-preview", "gemini-3.5-flash-lite-preview-06-17", "gemini-3.5-pro"]

    def __init__(self, clock):
        self.models = self
        self.clock = clock
        self.calls = {}

    @staticmethod
    def template_briefing(facts):
        s, why, act = facts["situation"], facts["why"], facts["action"]
        risk = why["risk_signals"]
        amount = f"${s['amount_usd']:,}" if s["amount_usd"] is not None else "no amount entered"
        reasons = " and ".join(f"{r['detail']} ({r['effect'].split()[0]} it)" for r in why["top_reasons"])
        signals = []  # up to two, most important first
        if risk["rep_much_more_optimistic_than_model"]:
            signals.append(f"the rep's forecast is {risk['rep_forecast_pct']}%, "
                           f"{risk['rep_forecast_minus_model_pct_points']} points above the model")
        if risk["close_date_slipped_times"] >= 2:
            signals.append(f"the close date has slipped {risk['close_date_slipped_times']} times")
        if risk["days_since_last_buyer_response"] == "never" or risk["days_since_last_buyer_response"] >= 21:
            signals.append(f"last buyer response: {risk['days_since_last_buyer_response']} days ago")
        if risk["stuck_in_stage"]:
            signals.append(f"it has been {risk['stuck_in_stage']}")
        signal = f", but {' and '.join(signals[:2])}" if signals else ""
        return {
            "situation": f"{facts['account_name']} is in {s['stage']} at {amount}, and the model's "
                         f"win probability is {s['model_win_probability_pct']}%.",
            "why": f"Main reasons: {reasons}{signal}.",
            "action": f"{act['next_best_action']}: {act['why_this_action']}.",
        }

    def list(self):
        names = self.MODELS + ["text-embedding-005"]
        return [SimpleNamespace(name=f"models/{n}", supported_actions=(
            ["embedContent"] if "embedding" in n else ["generateContent", "countTokens"])) for n in names]

    def generate_content(self, model, contents, config):
        self.calls[model] = self.calls.get(model, 0) + 1
        self.clock.sleep(2)  # a request takes a moment
        if model not in self.MODELS:
            raise errors.ClientError(404, {"error": {
                "code": 404, "status": "NOT_FOUND", "message": f"models/{model} is not found (fake)"}})
        if model == self.BUSY_MODEL:
            raise errors.ServerError(503, {"error": {
                "code": 503, "status": "UNAVAILABLE",
                "message": "The model is overloaded. (fake high demand)"}})
        body = self.template_briefing(json.loads(contents.split("Facts (JSON):", 1)[1]))
        return SimpleNamespace(
            text=json.dumps(body),
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
            usage_metadata=SimpleNamespace(prompt_token_count=1500, candidates_token_count=300,
                                           thoughts_token_count=200))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Write LLM deal briefings with Gemini.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be sent for the top deal; no API call")
    parser.add_argument("--fake", action="store_true",
                        help="use a built-in stand-in for Gemini (no key needed); test output only")
    parser.add_argument("--list-models", action="store_true",
                        help="list the models your key can use and which ones would be chosen")
    args = parser.parse_args()

    # ---- the client (real, fake, or none) ----
    if args.fake:
        clock = FakeClock()
        client = FakeGemini(clock)
    elif os.environ.get("GEMINI_API_KEY"):
        clock = Clock()
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1)))  # retries are handled here, not in the SDK
    else:
        clock, client = Clock(), None

    # ---- which models ----
    available = []
    if client is not None:
        try:
            available = list_models(client)
        except Exception as e:
            print(f"Could not list models ({type(e).__name__}: {e}); using built-in defaults.")
    main_model, fallback_model, how = choose_models(available)

    if args.list_models:
        if client is None:
            print("GEMINI_API_KEY is not set, so the list can't be fetched. "
                  "Set it first (or add --fake to see how the choice works).")
            return
        print(f"Models your key can use for text ({len(available)}):")
        for n in available:
            tag = "  <- main" if n == main_model else "  <- fallback" if n == fallback_model else ""
            print(f"  {n}{tag}")
        print(f"\nMain: {main_model}   Fallback: {fallback_model}   ({how})")
        return

    # ---- which deals need a (new) briefing ----
    briefings_file = BRIEFINGS_DIR / ("deal_briefings.fake.json" if args.fake else "deal_briefings.json")
    con = duckdb.connect(str(DB_PATH))
    data_through = con.execute("SELECT data_through FROM raw.export_info").fetchone()[0]
    deal_ids = select_deals(con)
    existing = {}
    if briefings_file.exists():
        try:
            existing = {b["deal_id"]: b for b in json.loads(briefings_file.read_text(encoding="utf-8"))}
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Warning: {briefings_file.name} is unreadable; starting without previous briefings.")
    plan = []
    for deal_id in deal_ids:
        facts = deal_facts(con, deal_id)
        key = material_key(facts)
        old = existing.get(deal_id)
        fresh = (old is not None and old.get("material_key") == key
                 and (data_through - date.fromisoformat(old["generated_on"])).days < REFRESH_AFTER_DAYS)
        plan.append((deal_id, facts, key, old if fresh else None))
    to_generate = sum(p[3] is None for p in plan)
    print(f"Briefings as of {data_through}: top {len(deal_ids)} deals, "
          f"{len(deal_ids) - to_generate} still up to date, {to_generate} to (re)write")
    print(f"Model: {main_model} (fallback: {fallback_model}) - {how}"
          + (" [FAKE stand-in]" if args.fake else ""))

    if args.dry_run:
        deal_id, facts, key, _ = plan[0]
        print(f"\n--- DRY RUN: request for deal {deal_id} (model {main_model}); nothing is sent ---\n")
        print("SYSTEM INSTRUCTION:\n" + SYSTEM_PROMPT + "\n")
        print("USER MESSAGE:\n" + user_message(facts))
        return
    if client is None and to_generate:
        print("GEMINI_API_KEY is not set: keeping existing briefings, writing no new ones. "
              "(Use --fake to test without a key.)")

    start = clock.now()
    briefings, stats = write_briefings(client, [main_model, fallback_model], plan, existing,
                                       data_through, clock, "fake" if args.fake else None)

    briefings_file.parent.mkdir(parents=True, exist_ok=True)
    briefings_file.write_text(json.dumps(briefings, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.fake:
        table = pd.DataFrame(briefings, columns=[
            "deal_id", "deal_name", "generated_on", "model", "material_key",
            "situation", "why", "action", "rep_vs_model", "stale", "stale_reason",
            "input_tokens", "output_tokens"])
        table["stale"] = table["stale"].fillna(False).astype(bool)
        save_table(con, "analytics.deal_briefings", table)
    con.close()

    used = ", ".join(f"{m}: {n}" for m, n in stats["by_model"].items()) or "none"
    print(f"\nSaved {len(briefings)} briefings to {briefings_file.relative_to(ROOT)} "
          f"({stats['written']} new, {stats['failed']} kept old / unavailable, "
          f"{sum(bool(b.get('stale')) for b in briefings)} marked stale) "
          f"in {clock.now() - start:.0f}s{' (virtual time)' if args.fake else ''}. "
          f"New briefings by model: {used}. Tokens: {stats['tokens_in']:,} in / {stats['tokens_out']:,} out.")


if __name__ == "__main__":
    main()
