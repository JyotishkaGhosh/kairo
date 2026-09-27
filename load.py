"""
load.py - copy the CRM export (data/raw/*.parquet) into DuckDB.

Creates data/kairo.duckdb with a `raw` schema: one table per Parquet file,
exactly as exported (no cleaning here - that is transform.py's job).
The database is rebuilt from the files on every run, so it always matches
the export. The hidden simulator truth in data/sim_state is never loaded.

Usage (PowerShell):
  python load.py
"""

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "kairo.duckdb"

TABLES = ["sales_reps", "accounts", "contacts", "leads", "deals",
          "deal_stage_history", "activities"]


def main():
    info_file = RAW_DIR / "export_info.json"
    if not info_file.exists():
        raise SystemExit("No CRM export found in data/raw. Run first:  python generate.py")
    data_through = json.loads(info_file.read_text())["data_through"]

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    print(f"Loading CRM export (data through {data_through}) into {DB_PATH.name}\n")
    print(f"{'table':<24}{'rows':>9}")
    for table in TABLES:
        path = (RAW_DIR / f"{table}.parquet").as_posix()
        con.execute(f"CREATE OR REPLACE TABLE raw.{table} AS SELECT * FROM read_parquet('{path}')")
        rows = con.execute(f"SELECT count(*) FROM raw.{table}").fetchone()[0]
        print(f"raw.{table:<20}{rows:>9,}")
    con.execute(f"""
        CREATE OR REPLACE TABLE raw.export_info AS
        SELECT DATE '{data_through}' AS data_through, current_timestamp AS loaded_at
    """)
    con.close()


if __name__ == "__main__":
    main()
