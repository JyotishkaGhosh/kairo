# Kairo — AI-powered CRM & sales intelligence

Resume project. The owner is a beginner: explain changes in simple words and
give Windows PowerShell commands (the project lives in `C:\projects\kairo`,
virtual environment in `.venv`).

## GIT RULE (most important rule in this repo)

**Never run `git commit`, `git push`, or anything else that creates commits or
changes git history** (no `git commit --amend`, `git rebase`, `git reset`,
`git merge`, `git tag`, `git stash`, `gh pr create`, etc.).
The owner makes every commit themselves in GitHub Desktop and must be the only
contributor on GitHub. Read-only git commands (`git status`, `git diff`,
`git log`) are fine. `.claude/settings.json` also blocks commit/push.

The one exception is `.github/workflows/daily.yml` (Milestone 8): on GitHub,
it commits the daily data refresh **under the owner's own identity**
("Jyotishka Ghosh" + 146670331+JyotishkaGhosh@users.noreply.github.com,
the same identity as GitHub Desktop; overridable with repo variables
GIT_AUTHOR_NAME / GIT_AUTHOR_EMAIL). Never a bot or Claude
identity, and never a `Co-Authored-By` line in that workflow's commit
message - the owner must stay the only contributor.

## Plan (build one milestone at a time, stop after each one)

1. **generate.py** — simulated B2B SaaS CRM data → `data/raw/*.parquet`  ✅
2. **load.py + transform.py** — load into DuckDB, clean data, build daily
   point-in-time deal snapshots  ✅
3. **Lead scoring model** (`lead_scoring.py`)  ✅
4. **Win probability + backtested revenue forecast** (`win_probability.py`,
   `revenue_forecast.py`)  ✅
5. **Customer segmentation + next-best-action** (`segmentation.py`,
   `next_best_action.py`)  ✅
6. **LLM-written deal briefings** (`deal_briefings.py`) — generated inside the
   pipeline; API key stored in GitHub Secrets (never in the repo)  ✅
7. **export.py** → `data.json` → static website on Vercel  ✅
8. **GitHub Actions** — run everything daily at 06:00 IST (00:30 UTC),
   committing under the owner's name (`run_pipeline.py`,
   `.github/workflows/daily.yml`)  ✅

## How the data works

- `python generate.py` — first run builds ~18 months of history ending
  yesterday. Every later run advances the CRM by exactly one day, continuing
  from saved state. It refuses to simulate a day in the future unless
  `--force` is given. `--reset` deletes generated data and starts over.
- Fixed seed (`SEED = 42`); each simulated day uses its own seeded random
  stream, so results are reproducible.
- `data/raw/` — the CRM tables a real company would export: `sales_reps`,
  `accounts`, `contacts`, `leads`, `deals`, `deal_stage_history`,
  `activities`.
- `data/sim_state/` — hidden ground truth (true win/convert probabilities,
  rep skill and optimism, planned outcomes) plus simulator bookkeeping.
  **Models and the website must never read this folder as input**; it is only
  for the simulator itself and for evaluating models afterwards.
- `deal_stage_history` logs every change to a deal (`change_type` =
  created / stage_change / close_date_change / owner_change) with the deal's
  stage, amount, probability and close date *at that moment* — this is what
  point-in-time snapshots are built from.
- `data/raw/export_info.json` holds `data_through` (last complete day). The
  pipeline takes "today" from here, never from the clock or `sim_state`.
- Parquet is written/read with DuckDB (no pyarrow needed).

## Pipeline: generate.py → load.py → transform.py

- `load.py` rebuilds `data/kairo.duckdb` (gitignored) schema `raw` from
  `data/raw/*.parquet`, unchanged.
- `transform.py` builds schema `clean` (duplicate contacts merged via
  `clean.contact_id_map`, missing categories → 'Unknown' / 'Not specified',
  helper columns) and `analytics.deal_daily_snapshots` (one row per open deal
  per day, state at END of that day). Then it runs data checks and exits with
  an error if any fail.
- Snapshot rule: every feature column must use only information dated on or
  before `snapshot_date`. `outcome_*` columns are future labels — for
  training targets only, never model inputs. The change log is ordered by
  `history_id` (not by timestamp) when rebuilding state.
- Account attributes (industry, size, region) are taken as currently known;
  they never change in this simulation.
- `analytics.lead_daily_snapshots` has no owner column: lead reassignments
  aren't logged, so a past owner is unknown. Current owner: `clean.leads`.

## Models

- `lead_scoring.py`: logistic regression on lead snapshots (gradient boosting
  is trained as a benchmark and only chosen if it cuts log loss by >1%).
  Backtest = train on what was known 120 days ago, test on later leads.
  Label rule: a lead still open 60 days after arrival counts as not converted
  (otherwise forgotten leads never get a label). Rows are weighted so each
  lead counts once. Writes `analytics.lead_scores` (open leads, score 0-100,
  grade A-D, top 3 reasons), `analytics.lead_score_drivers` and
  `analytics.model_metrics` (model = 'lead_scoring').
- `win_probability.py`: P(deal eventually won) from deal snapshots + the
  owner's point-in-time win rate. The rep-entered probability
  (`probability_pct`) is NOT a feature - it is kept only for comparison
  ("rep says X%, model says Y%"), the optimism risk flag and the backtest
  benchmark. Removing it left the backtest unchanged (AUC 0.852 → 0.854).
  Trains only on deals created 200+ days
  before the model date (outcome known; still open = lost), because quick
  closes are more often wins and would bias the model. Backtest: model as of
  90 days ago, tested on the next 90-day cohort, compared with the reps'
  probability field. Rows are weighted so each deal counts once (without it
  the model was overconfident). Writes `analytics.deal_scores` (model vs rep
  probability, rule-based risk flags, and `top_reasons`: JSON list of the 3
  `REASON_GROUPS` that move each deal's log-odds most vs the average
  training deal, each with factor / raises|lowers / strength / plain-English
  detail; computed from the logistic model, or a separate logistic
  explainer if boosting is chosen).
- P(won within 30 days) comes from a different model (revenue_forecast.py);
  wherever it is shown or used per deal (next_best_action.py, export.py) it
  is capped at the deal's overall win probability (`least(...)`) so the two
  numbers never contradict. Forecast totals use the raw value.
- `revenue_forecast.py`: bookings forecast for the next 30/90 days =
  Σ P(won within H days) × expected amount over open deals + new-pipeline
  estimate (past revenue per new lead × leads in the last H days). Label
  "won within H days" is complete for snapshots ≥ H days old, so no deal is
  dropped. Rows are NOT deal-weighted here (the forecast sums over every deal
  open on a day; weighting made the 30-day backtest worse). Walk-forward backtest every 14 days (each forecast rebuilt with
  only data known that day; needs ≥ 180 days of training history), compared
  with rep-weighted pipeline and run-rate. Range = ± the 80th-percentile
  backtest error. Uses the model kind chosen by win_probability.py.
  Writes `analytics.revenue_forecast`, `analytics.forecast_backtest`,
  `analytics.deal_close_forecast`.
- `segmentation.py`: K-means on customers (accounts with a won deal):
  log revenue, won deals, days since last win, tenure, 90-day activity and
  buyer responses, model-weighted open pipeline, size. N_SEGMENTS = 5 is
  fixed for day-to-day stability (silhouette for 3-7 is printed as a
  diagnostic). Names come from rules on each segment's profile
  (`name_segment`), ids ordered by total revenue. Writes
  `analytics.customer_segments`, `analytics.segment_profiles`.
- `next_best_action.py`: one action per open lead, open deal and customer,
  from transparent rules on model outputs (first matching rule wins).
  Deal rules, in order: close as lost (silent 90+ days or model < 2%) →
  re-engage (buyer silent 21+ days) → reset overdue close date → close plan
  (close date slipped 2+ times) → re-qualify (rep 25+ points above model) →
  push to close (P(won in 30 days) ≥ 60%) → unstick (stuck in stage) →
  follow up (no rep activity 14+ days) → decide if worth pursuing (model
  < 10%) → qualify (Prospecting / Qualified) → keep momentum ("on track";
  only if model ≥ 50% and rep < 15 points above it) → else strengthen the
  deal. Blocking problems come before "push to close" on purpose. The
  "rep says X%, model says Y%" comparison is its own column `rep_vs_model`,
  never inside `reason`. Counts in text use `ml_utils.plural` ("1 meeting").
  priority = value at stake ($ × model probability) × urgency (`URGENCY`).
  "hygiene" actions (close dead deals, drop stale leads, onboarding) have no
  $ value. It ranks where attention is worth most; it does NOT estimate
  causal uplift (the simulator's buyers don't react to rep actions). Also
  flags customers owned by reps who have left. Writes
  `analytics.next_best_actions` (rank_overall, rank_for_rep).
- `deal_briefings.py`: Google Gemini, free tier (`google-genai` SDK,
  `client.models.generate_content` with `response_mime_type="application/json"`
  + `response_json_schema`). For the top 20 deals by next-best-action
  priority it writes three short sentences - `situation` (stage, amount,
  model win probability), `why` (top 2 reasons from the win model's
  per-deal explanation + up to two risk signals: days since last activity /
  buyer response, close-date slips / passed, rep forecast ≥25 points above
  the model) and `action` (the next best action, made specific). The facts
  JSON sent to Gemini is grouped under exactly those three keys
  (`deal_facts`); the prompt allows only the numbers given, plain English,
  no hype. `PROMPT_VERSION` is part of the fingerprint: bump it whenever the
  prompt or output format changes so every briefing is rewritten. The site
  still renders the older headline/risks/next_steps format.
  - Models: each run lists the key's models (`client.models.list()`, those
    supporting generateContent) and picks the newest stable Flash-Lite as
    main and the next Flash-Lite (else newest Flash) as fallback, skipping
    preview/exp/tts/image/live/audio/embedding/thinking variants. Env
    `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` override. If listing fails:
    `DEFAULT_MODELS` (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`).
    Chosen models are printed; `--list-models` shows the list.
  - Resilience (Gemini often returns 503 "high demand"): SDK retries off
    (`HttpRetryOptions(attempts=1)`), 60 s timeout per request; 5 s pause
    between requests; per model retry after 5/15/30 s on 429/5xx/network
    errors; then the fallback model; a model that stayed unavailable (or
    404) is benched for the rest of the run. Invalid key (401/403/"API key")
    stops immediately. Whole step capped at `GEMINI_TIME_BUDGET_SECONDS`
    (default 300). Any deal not briefed keeps its previous briefing; the
    site shows "briefing unavailable" if none. The step never raises.
  - `run_pipeline.py` marks this step OPTIONAL: if it still exits non-zero,
    the pipeline warns and continues to export.py.
  Regenerates only when `material_key` (prompt version, stage, amount, close
  date, owner, action, win-prob band, top-reason factors, activity/response
  day bands, slips, optimism and stuck flags) changes or after 7 days. Stored in `data/briefings/deal_briefings.json` (committed) and
  `analytics.deal_briefings`; each briefing records the model that wrote it.
  Key from env `GEMINI_API_KEY` only. A briefing that should have been
  rewritten but wasn't (no key, time limit, busy models) is kept with
  `stale: true` + `stale_reason`; the site shows it as an older briefing.
  Every record also gets `rep_vs_model` from code (not from Gemini).
  `--dry-run` prints the request for the top deal without calling the API;
  `--fake` uses `FakeGemini` (realistic model list; `gemini-3.5-flash-lite`
  always answers 503; unknown names 404) with a virtual clock (no real
  waiting) and writes only `deal_briefings.fake.json` (gitignored), never
  the real briefings or DuckDB. Free-tier content may be used by Google to
  improve its products - acceptable only because all CRM data is simulated.
- `ml_utils.py`: shared model factories, fitting, evaluation, model
  comparison (keep logistic regression unless boosting cuts log loss >1%),
  calibration printout, `save_table`, `save_metrics`.
- Simulator realism: a deal's outcome is fixed at creation. Activity only
  hints at it: the win/loss differences live in `WIN_SIGNAL` in generate.py
  and are deliberately small, plus a random per-deal `engagement_level`
  (hidden, in deals_truth). An earlier, stronger version gave an unrealistic
  win-model AUC of ~0.93; now ~0.85. Don't strengthen these without reason.
- Don't tune models or forecast rules just to make a backtest look better;
  only make principled changes and say so.
- Models may read `data/sim_state` ONLY in clearly marked evaluation code
  (e.g. comparing with the hidden true odds), never for training or scoring.
- Pipeline order: generate.py → load.py → transform.py → lead_scoring.py →
  win_probability.py → revenue_forecast.py → segmentation.py →
  next_best_action.py → deal_briefings.py → export.py. The single source of
  this order is `STEPS` in `run_pipeline.py` (stops at the first failing
  step); add new steps there.

## Daily automation (`.github/workflows/daily.yml`)

- Cron `30 0 * * *` (06:00 IST) + manual `workflow_dispatch`; `TZ=Asia/Kolkata`
  so generate.py's "today" is the IST date. Python 3.11,
  `actions/checkout@v7`, `actions/setup-python@v7` (pip cache).
- Runs `python run_pipeline.py`; if any step fails (e.g. a data check), the
  job fails and nothing is committed.
- Secret `GEMINI_API_KEY` (briefings skipped if unset); optional repo
  variable `GEMINI_MODEL`.
- Commits `data/raw`, `data/sim_state`, `data/briefings`, `site/data.json`
  as "Daily data refresh: <date>" only if something changed; retries the
  push up to 3 times with `git pull --rebase`. The push triggers Vercel.
- The simulator's saved state lives in the repo, so the repo is the memory
  between runs: every run continues from the last committed day. A missed
  day is not caught up (each run advances exactly one day).

## Website (`site/`)

- `export.py` writes `site/data.json` (~30 KB, committed): KPIs, forecast +
  backtest + accuracy, monthly bookings, pipeline by stage, top 25 deals by
  next-best-action priority (with briefings from
  `data/briefings/deal_briefings.json`), top 15 leads + lead-score drivers,
  segments, actions, model metrics. Never exports `data/sim_state` or secrets.
- `site/index.html`, `styles.css`, `app.js`: static, no build step, no
  libraries. Charts are hand-built SVG following the dataviz skill: fixed
  series colors (Kairo model = slot 1 blue, reps = slot 2 orange, actual =
  slot 3 aqua; validated light + dark, all-pairs), 2px lines, ≤24px bars
  with 4px rounded data ends, hover/focus tooltips, "Show table" view on
  every chart, legend for ≥2 series. Colors are CSS tokens on `:root` with
  dark mode via `prefers-color-scheme` and the `data-theme` toggle.
  All data-derived text goes in via `textContent` (never `innerHTML`).
- Preview locally: `python -m http.server 8000 --directory site` →
  http://localhost:8000 (opening index.html from disk can't fetch data.json).
- Vercel: project Root Directory = `site`, Framework Preset = Other, no
  build command. Every push to GitHub (incl. the daily data commit) redeploys.
- Never write an API key into any file, log or commit. Never print it.

## Conventions

- Python 3.11, dependencies in `requirements.txt`.
- Keep scripts simple and readable; comment the "why", not the obvious.
