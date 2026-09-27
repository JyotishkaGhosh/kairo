# Kairo

An AI-powered CRM and sales-intelligence project, built on a realistic
simulated B2B SaaS sales pipeline that grows by one day, every day.

**Status:** all 8 milestones built — data simulator, DuckDB pipeline, lead
scoring, win probability, a backtested revenue forecast, customer
segmentation, next-best-action recommendations, LLM-written deal briefings,
a static website and a daily automated refresh.

## What's here so far

`generate.py` simulates a software company's CRM: sales reps, accounts,
contacts, leads, deals moving through stages
(Prospecting → Qualified → Demo → Proposal → Negotiation → Closed Won / Lost),
the full history of deal changes, and calls, emails and meetings.

Every deal has a hidden "true" chance of winning, driven by industry, company
size, lead source, engagement and rep skill, so later ML models have real
patterns to find. The data is also messy on purpose: close dates slip, deals
go quiet, some reps are over-optimistic, fields are missing and a few
contacts are duplicated.

`load.py` copies the export into a DuckDB database, and `transform.py`
cleans it (merging duplicate contacts, labelling missing values) and builds
**daily point-in-time deal snapshots**: for every day, what each open deal
looked like on that day, using only what was known then. That is what makes
honest model training and forecast backtesting possible. The pipeline ends
with automatic data-quality checks.

`lead_scoring.py` scores every open lead from 0 to 100 (grade A–D) with the
top reasons behind the score. It is backtested honestly: built as if it were
120 days ago, then tested on the leads that arrived afterwards. In that test
the top-scored 20% of new leads converted about twice as often as average,
and the predicted percentages matched real conversion rates closely.

`win_probability.py` gives every open deal a model win probability next to
the rep's own guess, plus risk flags (overdue close date, buyer gone quiet,
stuck in stage). The rep's guess is not a model input - it is only shown for
comparison ("rep says 80%, model says 49%"). In the backtest the reps' probabilities were too
optimistic (deals they rated 84% on average were won 65% of the time); the
model's predictions matched reality within a few points.

`revenue_forecast.py` forecasts revenue won in the next 30 and 90 days, with
a range. Its walk-forward backtest rebuilds every past forecast using only
data known on that day:

| average error | next 30 days | next 90 days |
|---|---|---|
| Kairo forecast | ~20% | ~14% |
| run-rate ("same as last period") | ~45% | ~24% |
| rep-weighted pipeline (typical CRM) | ~234% | ~67% |

`segmentation.py` groups customers with K-means into five named segments
(e.g. *High-value – quiet*, *New customers – engaged*, *Repeat buyers*).

`next_best_action.py` turns all of this into a to-do list: one recommended
action per lead, deal and customer ("Call today – the buyer is responding",
"Re-engage the buyer", "Agree a close plan with the buyer", "Push to close
this month", "Qualify the opportunity", "Pitch an expansion", "Close as
lost"), each with the reason and the dollars at stake, ranked per
rep. The rules are transparent and built on the model outputs; they rank
where attention is worth the most, rather than claiming to predict the
effect of an action.

`deal_briefings.py` asks Google Gemini (a free-tier Flash-Lite model, picked
automatically from the models the API key can use, with a fallback model) to
write a three-sentence briefing for each of the 20 most important deals:
**situation** (stage, amount, the model's win probability), **why** (the win
model's top two reasons for this deal plus up to two risk signals) and
**action** (the recommended next step). Gemini only uses numbers the pipeline
computed (it is told never to invent anything) and returns structured JSON. Briefings are only regenerated when something material
about a deal changes. When Gemini is busy, the script retries, switches to the
fallback model, stops after 5 minutes at most and keeps each deal's previous
briefing, so a busy AI service can never break the daily refresh. The API key comes from an environment variable / GitHub
Secret and is never stored in the repository.

`export.py` gathers the results into one small `site/data.json`, and the
static website in `site/` (plain HTML, CSS and JavaScript with hand-built
SVG charts, light and dark mode, works on phones) displays it. It is hosted
on Vercel, which republishes the site whenever the repository changes.

Every morning at 06:00 IST a GitHub Actions workflow
(`.github/workflows/daily.yml`) advances the simulated CRM by one day, runs
the whole pipeline (`run_pipeline.py`), commits the new data and pushes it —
which updates the website. If any data check fails, nothing is published.

## Run it (Windows PowerShell)

```powershell
cd C:\projects\kairo
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py      # everything below, in order (about 1 minute)

# or step by step:
python generate.py          # 1st run: ~18 months of history. Later runs: +1 day.
python load.py              # Parquet -> data/kairo.duckdb (schema raw)
python transform.py         # cleaning + snapshots (schemas clean, analytics) + checks
python lead_scoring.py      # backtest + score today's open leads
python win_probability.py   # backtest + win probability for today's open deals
python revenue_forecast.py  # walk-forward backtest + 30/90-day revenue forecast
python segmentation.py      # customer segments
python next_best_action.py  # ranked to-do list per rep
python deal_briefings.py    # Gemini briefings (needs $env:GEMINI_API_KEY; --fake / --dry-run work without)
python deal_briefings.py --list-models   # which Gemini models your key can use, and which are chosen
python export.py            # results -> site/data.json

# preview the website at http://localhost:8000
python -m http.server 8000 --directory site
python generate.py --reset  # throw away generated data and rebuild
```

Output goes to `data/raw/*.parquet` (the CRM tables) and `data/sim_state/`
(hidden ground truth used only by the simulator and for model evaluation).

## Roadmap

1. Simulated CRM data ✅
2. DuckDB loading, cleaning, daily point-in-time deal snapshots ✅
3. Lead scoring model ✅
4. Win probability + backtested revenue forecast ✅
5. Customer segmentation + next-best-action ✅
6. LLM-written deal briefings ✅
7. Static website (Vercel) ✅
8. Daily automated refresh with GitHub Actions ✅
