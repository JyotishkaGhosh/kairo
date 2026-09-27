# Kairo

An AI-powered CRM and sales-intelligence project, built on a realistic
simulated B2B SaaS sales pipeline that grows by one day, every day.

**Status:** Milestone 6 of 8 — data simulator, DuckDB pipeline, lead scoring,
win probability, a backtested revenue forecast, customer segmentation,
next-best-action recommendations and LLM-written deal briefings.

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
stuck in stage). In the backtest the reps' probabilities were too
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
"Re-engage the buyer", "Push to close this month", "Pitch an expansion",
"Close as lost"), each with the reason and the dollars at stake, ranked per
rep. The rules are transparent and built on the model outputs; they rank
where attention is worth the most, rather than claiming to predict the
effect of an action.

`deal_briefings.py` asks Claude (Anthropic API) to turn the facts about each of
the 20 most important deals into a short briefing: headline, situation, risks
and next steps. Claude only rewrites facts the pipeline computed (it is told
never to invent anything) and returns structured JSON. Briefings are only
regenerated when something material about a deal changes, which keeps API
costs low. The API key comes from an environment variable / GitHub Secret and
is never stored in the repository.

## Run it (Windows PowerShell)

```powershell
cd C:\projects\kairo
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate.py          # 1st run: ~18 months of history. Later runs: +1 day.
python load.py              # Parquet -> data/kairo.duckdb (schema raw)
python transform.py         # cleaning + snapshots (schemas clean, analytics) + checks
python lead_scoring.py      # backtest + score today's open leads
python win_probability.py   # backtest + win probability for today's open deals
python revenue_forecast.py  # walk-forward backtest + 30/90-day revenue forecast
python segmentation.py      # customer segments
python next_best_action.py  # ranked to-do list per rep
python deal_briefings.py    # LLM briefings (needs $env:ANTHROPIC_API_KEY; --dry-run works without)
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
7. Static website (Vercel)
8. Daily automated refresh with GitHub Actions
