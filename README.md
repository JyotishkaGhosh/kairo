# Kairo

**AI-powered CRM and sales intelligence**, built on a realistic simulated
B2B SaaS sales pipeline that grows by one day, every day.

**Live site:** https://kairo-five-nu.vercel.app/

**Author:** Jyotishka Ghosh

---

## What Kairo is

Kairo takes the kind of data a sales team keeps in its CRM (accounts,
contacts, leads, deals, calls, emails, meetings) and turns it into answers a
sales team actually needs:

- **Which leads should we call first?** A lead score (0-100, grade A-D) with
  the top reasons behind it.
- **Which deals will we really win?** A model win probability for every open
  deal, shown next to the rep's own guess ("rep says 80%, model says 49%"),
  plus risk flags (overdue close date, buyer gone quiet, stuck in stage).
- **How much revenue is coming?** A 30- and 90-day bookings forecast with a
  range, checked against history with a walk-forward backtest.
- **Who are our customers?** Five named customer segments.
- **What should each rep do next?** One recommended action per lead, deal
  and customer, ranked by the money at stake.
- **What's the story of each big deal?** A three-sentence AI briefing
  (situation, why, action) for the 20 most important deals, written by
  Google Gemini from the pipeline's own numbers only.

Everything is refreshed automatically every morning and published as a
static website.

The CRM data is **simulated** (no real company or person), but made
realistic on purpose: every deal has a hidden "true" chance of winning,
close dates slip, deals go quiet, some reps are over-optimistic, fields are
missing and some contacts are duplicated. Because the true odds are known,
the models can be checked honestly.

## Architecture

```
                 GitHub Actions, every day 06:00 IST (run_pipeline.py)
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 │  generate.py ──► data/raw/*.parquet        (CRM export, +1 simulated day)  │
 │       │          data/sim_state/           (hidden truth: simulator only)  │
 │       ▼                                                                    │
 │  load.py ──────► DuckDB  schema raw                                        │
 │       ▼                                                                    │
 │  transform.py ─► schema clean + analytics.deal/lead_daily_snapshots        │
 │       │          (point-in-time: only what was known that day) + checks    │
 │       ▼                                                                    │
 │  lead_scoring.py ──────► lead scores + reasons                             │
 │  win_probability.py ───► deal win probability + reasons + risk flags       │
 │  revenue_forecast.py ──► 30/90-day forecast + walk-forward backtest        │
 │  segmentation.py ──────► customer segments (K-means)                       │
 │  next_best_action.py ──► one ranked action per lead / deal / customer      │
 │       ▼                                                                    │
 │  deal_briefings.py ───► Google Gemini ──► data/briefings/*.json            │
 │       │                 (optional step: a busy AI never stops the run)     │
 │       ▼                                                                    │
 │  export.py ────► site/data.json                                            │
 │                                                                            │
 └──────────────┬─────────────────────────────────────────────────────────────┘
                │ git commit (as the author) + push
                ▼
          GitHub repo ──► Vercel redeploys ──► static site: site/index.html
                                                  + app.js reads data.json
```

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data storage & SQL | DuckDB, Parquet (read and written by DuckDB) |
| Data wrangling | pandas, NumPy |
| Machine learning | scikit-learn (logistic regression, gradient boosting benchmark, K-means) |
| Simulated data | Faker + a custom day-by-day CRM simulator |
| AI briefings | Google Gemini API (free-tier Flash-Lite, `google-genai` SDK, structured JSON output) |
| Website | Plain HTML, CSS and JavaScript; hand-built SVG charts; light and dark mode; no build step |
| Automation | GitHub Actions (daily cron), GitHub Secrets for the API key |
| Hosting | Vercel (serves the `site/` folder) |

## How the pipeline works

1. **Simulate** (`generate.py`). The first run builds about 18 months of
   history. Every later run advances the CRM by exactly one day, continuing
   from saved state, with a fixed random seed so results are reproducible.
2. **Load and clean** (`load.py`, `transform.py`). The export is copied into
   DuckDB and cleaned: duplicate contacts are merged and missing categories
   are labelled. Then the step builds **daily point-in-time snapshots**, one
   row per open deal (and lead) per day, using only information known on
   that day. This is what makes honest training and backtesting possible.
   Automatic data checks stop the pipeline if anything looks wrong.
3. **Score leads** (`lead_scoring.py`). Logistic regression on lead
   snapshots, with a gradient-boosting benchmark that is only chosen if it
   is clearly better.
4. **Win probability** (`win_probability.py`). Logistic regression on deal
   snapshots. The rep's own probability is **not** a model input: it is kept
   only for comparison, so the model stands on its own evidence. Each deal
   gets its top reasons, e.g. "7 buyer responses in the last 30 days raises
   the win probability".
5. **Revenue forecast** (`revenue_forecast.py`). The chance of each open
   deal being won within 30 / 90 days times its expected amount, plus an
   estimate for deals not created yet.
6. **Segments** (`segmentation.py`). K-means on customers (revenue,
   recency, activity, open pipeline, size), five named segments.
7. **Next best action** (`next_best_action.py`). Transparent rules on the
   model outputs. Blocking problems come first (buyer silent, close date
   passed or slipping, rep far more optimistic than the model); only then
   "push to close" (60%+ chance within 30 days), "on track" (model 50%+ and
   rep not 15+ points above it), and so on. Ranked by value at stake ×
   urgency. It ranks where attention is worth the most; it does not claim to
   predict the effect of an action.
8. **AI briefings** (`deal_briefings.py`). Gemini writes three sentences per
   top deal from a JSON of facts, and is told to use only the numbers given.
   A briefing is only rewritten when something material changes. Retries,
   a fallback model and a time limit mean a busy AI service can never break
   the daily refresh. Any briefing that couldn't be rewritten is kept and
   marked `stale`, with the reason.
9. **Export and publish** (`export.py`, `.github/workflows/daily.yml`). The
   results go into one small `site/data.json`. The daily workflow commits
   the new data under the author's own name and pushes it, and Vercel
   redeploys the site.

## Model results

Every model is **backtested**: rebuilt as if it were an earlier date, using
only data known then, and tested on what happened afterwards. The numbers
below are from the run of 2026-09-27; they move slightly every day as the
simulated company grows.

### Win probability (deals)

Built as of 2026-06-29 and tested on the next 195 deals.

| | AUC (higher is better) | Log loss (lower is better) | Brier (lower is better) |
|---|---|---|---|
| **Kairo model** | **0.854** | **0.423** | **0.137** |
| Reps' own probabilities | 0.714 | 0.573 | 0.193 |

Calibration (when the model says X%, how often is the deal really won?):

| Model score | Predicted | Actually won |
|---|---|---|
| 0-10% | 4% | 5% |
| 10-20% | 14% | 18% |
| 20-30% | 25% | 29% |
| 30-45% | 37% | 40% |
| 45-70% | 56% | 58% |
| 70-90% | 80% | 81% |
| 90%+ | 95% | 96% |

The model's predictions match reality within a few points. The reps' own
numbers are too optimistic: deals they rated 84% on average were won 65% of
the time. One caveat at the top end: of the 48 deals that reached 90%+ on
at least one day, 41 were won (85%), because some deals scored high only
briefly.

### Lead scoring

Built as of 2026-05-30 and tested on the next 635 leads. AUC 0.734 (log
loss 0.465 vs 0.532 for "everyone gets the average"). The top-scored 20% of
new leads converted at 41%, vs 22% on average (1.8×).

| Lead score | Predicted | Actually converted |
|---|---|---|
| 0-10% | 7% | 5% |
| 10-20% | 15% | 14% |
| 20-30% | 25% | 23% |
| 30-45% | 37% | 30% |
| 45%+ | 58% | 52% |

### Revenue forecast

Walk-forward backtest: every past forecast rebuilt with only the data known
on that day (22 forecasts for 30 days, 13 for 90 days).

| Average error | Next 30 days | Next 90 days |
|---|---|---|
| **Kairo forecast** | **20%** | **16%** |
| Run-rate ("same as last period") | 45% | 24% |
| Rep-weighted pipeline (typical CRM) | 234% | 67% |

In 80% of past forecasts, actual revenue was within ±32% (30 days) and ±22%
(90 days) of the Kairo forecast. These ranges are what the site shows.

## Run it locally (Windows PowerShell)

Requirements: Python 3.11.

```powershell
cd C:\projects\kairo
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_pipeline.py
.venv\Scripts\python.exe -m http.server 8000 --directory site
```

Then open http://localhost:8000. The site must be served like this; opening
`index.html` straight from disk can't load `data.json`.

AI briefings need a free Google AI Studio API key. Without one, the pipeline
still runs and keeps the existing briefings.

```powershell
$env:GEMINI_API_KEY = "paste-your-key-here"
.venv\Scripts\python.exe deal_briefings.py
.venv\Scripts\python.exe export.py
```

Useful extras:

```powershell
.venv\Scripts\python.exe deal_briefings.py --dry-run      # show what would be sent to Gemini, no API call
.venv\Scripts\python.exe deal_briefings.py --fake         # test run with a stand-in for Gemini, no key needed
.venv\Scripts\python.exe deal_briefings.py --list-models  # models your key can use
.venv\Scripts\python.exe generate.py --reset              # throw away generated data and start over
```

Each step can also be run on its own, in this order: `generate.py`,
`load.py`, `transform.py`, `lead_scoring.py`, `win_probability.py`,
`revenue_forecast.py`, `segmentation.py`, `next_best_action.py`,
`deal_briefings.py`, `export.py`.

## Deployment

- **Website:** Vercel project with Root Directory = `site`, Framework
  Preset = Other, no build command. The page (`site/index.html` +
  `site/app.js`) loads `site/data.json` from the same folder.
- **Daily refresh:** `.github/workflows/daily.yml` runs at 06:00 IST (and on
  demand from the Actions tab). The Gemini key is the repository secret
  `GEMINI_API_KEY` and is never stored in the repository. The data commit
  is made as Jyotishka Ghosh (GitHub noreply email), never by a bot.

## Project layout

```
generate.py … export.py   pipeline steps (see above); run_pipeline.py runs them in order
ml_utils.py               shared model, evaluation and saving helpers
data/raw/                 simulated CRM export (Parquet)
data/sim_state/           simulator's hidden truth, never used as model input
data/briefings/           AI briefings (JSON)
site/                     the website: index.html, styles.css, app.js, data.json
.github/workflows/        daily automation
```

## Author

Made by **Jyotishka Ghosh**.

© 2026 Jyotishka Ghosh. All rights reserved.
