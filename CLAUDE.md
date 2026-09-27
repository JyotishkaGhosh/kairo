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

The one planned exception is Milestone 8: a GitHub Actions workflow that
commits the daily data refresh **under the owner's own name and email**
(never a bot or Claude identity).

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
7. **export.py** → `data.json` → static website on Vercel
8. **GitHub Actions** — run everything daily at 06:00 IST (00:30 UTC),
   committing under the owner's name

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
  owner's point-in-time win rate. Trains only on deals created 200+ days
  before the model date (outcome known; still open = lost), because quick
  closes are more often wins and would bias the model. Backtest: model as of
  90 days ago, tested on the next 90-day cohort, compared with the reps'
  probability field. Rows are weighted so each deal counts once (without it
  the model was overconfident). Writes `analytics.deal_scores` (model vs rep
  probability, rule-based risk flags).
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
  priority = value at stake ($ × model probability) × urgency (`URGENCY`).
  "hygiene" actions (close dead deals, drop stale leads, onboarding) have no
  $ value. It ranks where attention is worth most; it does NOT estimate
  causal uplift (the simulator's buyers don't react to rep actions). Also
  flags customers owned by reps who have left. Writes
  `analytics.next_best_actions` (rank_overall, rank_for_rep).
- `deal_briefings.py`: Claude API (`anthropic` SDK, model `claude-opus-5`,
  effort medium, structured JSON output via `output_config.format`,
  refusal fallback `fallbacks="default"` + beta
  `server-side-fallback-2026-07-01`) writes headline / situation / risks /
  next_steps for the top 20 deals by next-best-action priority, from a facts
  JSON only (prompt forbids inventing anything; absolute dates only).
  Regenerates only when `material_key` (stage, amount, close date, owner,
  action, risk flags, meetings, pushes, win-prob band) changes or after 7
  days. Stored in `data/briefings/deal_briefings.json` (committed) and
  `analytics.deal_briefings`. Key from env `ANTHROPIC_API_KEY` only; without
  it the script keeps existing briefings and exits cleanly.
  `--dry-run` prints the request for the top deal without calling the API.
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
  next_best_action.py → deal_briefings.py
- Never write an API key into any file, log or commit. Never print it.

## Conventions

- Python 3.11, dependencies in `requirements.txt`.
- Keep scripts simple and readable; comment the "why", not the obvious.
