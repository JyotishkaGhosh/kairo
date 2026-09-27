"""
run_pipeline.py - Milestone 8: run the whole Kairo pipeline, in order.

Each step is its own script (see CLAUDE.md). This runs them one after another
with the same Python, stops at the first failure (so a failed data check
never reaches the models or the website), and prints how long each step took.
GitHub Actions runs exactly this every day; you can run it locally too.

Usage (PowerShell):
  python run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED, OPTIONAL = True, False
STEPS = [
    ("Advance the simulated CRM by one day", "generate.py", REQUIRED),
    ("Load the CRM export into DuckDB", "load.py", REQUIRED),
    ("Clean, snapshot and check the data", "transform.py", REQUIRED),
    ("Score leads", "lead_scoring.py", REQUIRED),
    ("Win probability for open deals", "win_probability.py", REQUIRED),
    ("Revenue forecast + backtest", "revenue_forecast.py", REQUIRED),
    ("Customer segments", "segmentation.py", REQUIRED),
    ("Next best actions", "next_best_action.py", REQUIRED),
    # Optional: an outside service (Gemini) must never stop the daily refresh.
    # If it fails, the previous briefings stay in place.
    ("AI deal briefings (Gemini)", "deal_briefings.py", OPTIONAL),
    ("Export site/data.json", "export.py", REQUIRED),
]


def main():
    started = time.time()
    warnings = []
    for number, (title, script, required) in enumerate(STEPS, 1):
        print(f"\n=== [{number}/{len(STEPS)}] {title} ({script}) ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0 and required:
            print(f"\nFAILED at step {number} ({script}), exit code {result.returncode}. "
                  f"Later steps were not run.", flush=True)
            sys.exit(result.returncode)
        if result.returncode != 0:
            warnings.append(script)
            print(f"--- WARNING: optional step {script} failed (exit code {result.returncode}); "
                  f"continuing with the previous results.", flush=True)
            continue
        print(f"--- done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nPipeline finished in {time.time() - started:.0f}s"
          + (f", with warnings from: {', '.join(warnings)}." if warnings else "."))


if __name__ == "__main__":
    main()
