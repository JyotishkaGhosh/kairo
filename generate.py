"""
generate.py - Kairo's simulated B2B SaaS CRM.

First run : builds ~18 months of CRM history, ending yesterday.
Later runs: loads the saved CRM and advances it by exactly one day.

Outputs (Parquet):
  data/raw/        the CRM tables a real company would have
  data/sim_state/  hidden ground truth (true win chances, rep skill, planned
                   outcomes) + simulator bookkeeping. Models must NOT use it
                   as input; it is only here to continue the simulation and
                   to check later how good the models really are.

Usage (PowerShell):
  python generate.py                         # build history, or add one day
  python generate.py --reset                 # delete generated data, rebuild
  python generate.py --end-date 2026-09-26   # choose where history ends (first run only)
  python generate.py --force                 # allow simulating a future day
"""

import argparse
import json
import math
import os
import shutil
import unicodedata
from datetime import date, datetime, time, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
HISTORY_DAYS = 548  # ~18 months

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
STATE_DIR = ROOT / "data" / "sim_state"
STATE_FILE = STATE_DIR / "state.json"

# ---------------------------------------------------------------------------
# Table schemas (column -> type). Used to write Parquet with exact types.
# ---------------------------------------------------------------------------
RAW_SCHEMAS = {
    "sales_reps": {
        "rep_id": "int", "name": "str", "email": "str", "region": "str",
        "hire_date": "date", "termination_date": "date", "is_active": "bool",
        "annual_quota_usd": "float",
    },
    "accounts": {
        "account_id": "int", "account_name": "str", "domain": "str",
        "industry": "str", "employee_count": "int", "size_band": "str",
        "region": "str", "country": "str", "annual_revenue_usd": "float",
        "owner_rep_id": "int", "created_at": "datetime",
    },
    "contacts": {
        "contact_id": "int", "account_id": "int", "first_name": "str",
        "last_name": "str", "email": "str", "phone": "str", "job_title": "str",
        "seniority": "str", "created_at": "datetime",
    },
    "leads": {
        "lead_id": "int", "created_at": "datetime", "source": "str",
        "contact_id": "int", "account_id": "int", "owner_rep_id": "int",
        "status": "str", "status_changed_at": "datetime",
        "disqualify_reason": "str", "converted_deal_id": "int",
    },
    "deals": {
        "deal_id": "int", "deal_name": "str", "account_id": "int",
        "primary_contact_id": "int", "lead_id": "int", "owner_rep_id": "int",
        "deal_type": "str", "source": "str", "product_tier": "str",
        "stage": "str", "amount_usd": "float", "probability_pct": "int",
        "expected_close_date": "date", "created_at": "datetime",
        "stage_changed_at": "datetime", "closed_at": "datetime",
        "is_closed": "bool", "is_won": "bool", "lost_reason": "str",
    },
    "deal_stage_history": {
        "history_id": "int", "deal_id": "int", "changed_at": "datetime",
        "change_type": "str", "from_stage": "str", "to_stage": "str",
        "amount_usd": "float", "probability_pct": "int",
        "expected_close_date": "date", "owner_rep_id": "int",
    },
    "activities": {
        "activity_id": "int", "activity_at": "datetime", "activity_type": "str",
        "direction": "str", "outcome": "str", "duration_min": "int",
        "rep_id": "int", "contact_id": "int", "account_id": "int",
        "lead_id": "int", "deal_id": "int",
    },
}

TRUTH_SCHEMAS = {
    "reps_truth": {
        "rep_id": "int", "skill": "float", "optimism": "float",
        "activity_level": "float",
    },
    "accounts_truth": {
        "account_id": "int", "true_industry": "str", "true_size_band": "str",
        "name_locale": "str",
    },
    "contacts_truth": {"contact_id": "int", "true_seniority": "str"},
    "leads_truth": {
        "lead_id": "int", "true_convert_prob": "float", "will_convert": "bool",
        "first_touch_date": "date", "resolve_date": "date", "replies": "int",
    },
    "deals_truth": {
        "deal_id": "int", "true_win_prob": "float", "will_win": "bool",
        "final_stage": "str", "true_amount": "float", "discount": "float",
        "cycle_mult": "float", "engagement_level": "float",
        "next_event_date": "date", "dark_since": "date",
    },
}
ID_COLUMN = {
    "sales_reps": "rep_id", "accounts": "account_id", "contacts": "contact_id",
    "leads": "lead_id", "deals": "deal_id", "deal_stage_history": "history_id",
    "activities": "activity_id", "reps_truth": "rep_id",
    "accounts_truth": "account_id", "contacts_truth": "contact_id",
    "leads_truth": "lead_id", "deals_truth": "deal_id",
}
SQL_TYPES = {"int": "BIGINT", "float": "DOUBLE", "str": "VARCHAR",
             "date": "DATE", "datetime": "TIMESTAMP", "bool": "BOOLEAN"}

# ---------------------------------------------------------------------------
# Business assumptions. "effect" values are added to a log-odds score:
# positive = more likely to convert / win, negative = less likely.
# ---------------------------------------------------------------------------
REGIONS = {  # weight, initial reps, [(country, faker locale, weight)]
    "North America": (0.45, 8, [("United States", "en_US", 0.85), ("Canada", "en_CA", 0.15)]),
    "EMEA": (0.30, 6, [("United Kingdom", "en_GB", 0.35), ("Germany", "de_DE", 0.25),
                       ("France", "fr_FR", 0.15), ("Netherlands", "nl_NL", 0.10),
                       ("Spain", "es_ES", 0.08), ("Sweden", "sv_SE", 0.07)]),
    "APAC": (0.15, 4, [("India", "en_IN", 0.40), ("Australia", "en_AU", 0.35),
                       ("Singapore", "en_US", 0.15), ("New Zealand", "en_NZ", 0.10)]),
    "LATAM": (0.10, 3, [("Brazil", "pt_BR", 0.55), ("Mexico", "es_MX", 0.30),
                        ("Chile", "es_MX", 0.15)]),
}
INDUSTRIES = {  # weight, win effect, sales-cycle multiplier
    "Software": (0.20, 0.45, 1.0), "Fintech": (0.12, 0.20, 1.1),
    "Healthcare": (0.12, -0.20, 1.3), "Retail": (0.10, -0.05, 0.9),
    "Manufacturing": (0.12, -0.30, 1.2), "Education": (0.08, -0.45, 1.2),
    "Media": (0.08, 0.05, 0.9), "Logistics": (0.10, -0.10, 1.0),
    "Government": (0.08, -0.70, 1.6),
}
SIZE_BANDS = {  # weight, win effect, cycle multiplier, employee range, deal size multiplier
    "SMB": (0.50, 0.30, 0.6, (10, 199), 0.8),
    "Mid-Market": (0.35, 0.00, 1.0, (200, 1999), 1.0),
    "Enterprise": (0.15, -0.45, 1.7, (2000, 60000), 1.6),
}
SOURCES = {  # weight, effect
    "inbound": (0.40, 0.35), "outbound": (0.30, -0.55),
    "referral": (0.12, 0.80), "event": (0.18, 0.00),
}
SENIORITY = {  # weight, effect, example titles
    "C-level": (0.08, 0.30, ["CEO", "CTO", "CFO", "COO", "Chief Revenue Officer"]),
    "VP": (0.15, 0.30, ["VP of Sales", "VP of Engineering", "VP of Operations", "VP of Marketing"]),
    "Director": (0.27, 0.10, ["Director of IT", "Director of Sales Operations", "Director of Customer Success", "Head of Revenue Operations"]),
    "Manager": (0.32, 0.00, ["Sales Manager", "IT Manager", "Operations Manager", "Marketing Manager"]),
    "Individual Contributor": (0.18, -0.45, ["Sales Operations Analyst", "Business Analyst", "Account Executive", "Systems Administrator"]),
}
PRODUCT_TIERS = {"Starter": 8_000, "Growth": 30_000, "Enterprise": 110_000}  # typical annual contract value
TIER_MIX = {"SMB": [0.60, 0.35, 0.05], "Mid-Market": [0.20, 0.55, 0.25], "Enterprise": [0.02, 0.33, 0.65]}

STAGES = ["Prospecting", "Qualified", "Demo", "Proposal", "Negotiation"]
WON, LOST = "Closed Won", "Closed Lost"
STAGE_PROB = [10, 20, 35, 55, 75]         # default probability a rep enters per stage
STAGE_DAYS = [13, 16, 19, 22, 16]        # mean days in each stage (Mid-Market)
LOSS_STAGE_WEIGHTS = [0.30, 0.27, 0.20, 0.13, 0.10]  # where losing deals die
ACT_RATE = [0.35, 0.45, 0.55, 0.60, 0.65]  # activities per deal per workday
ACT_MIX = [[0.60, 0.35, 0.05], [0.50, 0.30, 0.20], [0.40, 0.25, 0.35],
           [0.50, 0.20, 0.30], [0.50, 0.25, 0.25]]  # email / call / meeting
LOST_REASONS = ["Lost to competitor", "No budget", "Price too high",
                "Chose to build in-house", "Bad timing", "Missing features"]
DISQUALIFY_REASONS = ["Not a fit", "No budget", "No response", "Using competitor", "Student / job seeker"]

# How winning deals differ from losing ones in day-to-day activity:
# (value if the deal will be won, value if it will be lost). Kept small on
# purpose - in real life engagement hints at the outcome, it doesn't give it
# away. On top of this, every deal gets its own random "engagement level"
# (some buyers are simply chattier), which blurs the signal further.
WIN_SIGNAL = {
    "activity": (1.07, 0.95),      # multiplier on the number of activities
    "inbound_email": (0.36, 0.30),  # share of emails that are buyer replies
    "call_connect": (0.60, 0.54),   # share of calls that reach someone
    "meeting_held": (0.92, 0.88),   # share of meetings that happen (vs no-show)
    "stage_speed": (0.95, 1.05),    # multiplier on time spent per stage
}
ENGAGEMENT_SPREAD = 0.4   # spread of the per-deal engagement level (log-normal sigma)
LOSING_COOL_OFF = 0.7     # activity multiplier for a losing deal in its last stage

LEAD_BASE = -1.45         # base log-odds that a lead converts to a deal
DEAL_BASE = -2.10          # base log-odds that a new-business deal is won
EXPANSION_EFFECT = 1.10    # existing customers buy more easily
LEADS_PER_WORKDAY = 11.0
MONTH_FACTOR = {12: 0.75, 8: 0.85, 1: 0.90}  # holiday / summer dips


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def slug(text):
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def is_workday(day):
    return day.weekday() < 5


def next_workday(day):
    while not is_workday(day):
        day += timedelta(days=1)
    return day


# ---------------------------------------------------------------------------
# Parquet helpers (DuckDB does the writing, so pyarrow is not required)
# ---------------------------------------------------------------------------
def write_parquet(path, rows, schema):
    df = pd.DataFrame({col: pd.Series([r.get(col) for r in rows], dtype=object) for col in schema})
    select = ", ".join(f'CAST("{c}" AS {SQL_TYPES[t]}) AS "{c}"' for c, t in schema.items())
    tmp = path.with_suffix(".parquet.tmp")
    con = duckdb.connect()
    con.register("df", df)
    con.execute(f"COPY (SELECT {select} FROM df) TO '{tmp.as_posix()}' (FORMAT PARQUET)")
    con.close()
    os.replace(tmp, path)  # swap in only after a complete write


def read_parquet(path):
    con = duckdb.connect()
    result = con.execute("SELECT * FROM read_parquet(?)", [path.as_posix()])
    cols = [c[0] for c in result.description]
    rows = [dict(zip(cols, row)) for row in result.fetchall()]
    con.close()
    return rows


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class CRMSimulator:
    def __init__(self, state):
        self.state = state  # seed, dates, next ids, pending hires
        self.rows = {name: [] for name in RAW_SCHEMAS}
        self.truth = {name: {} for name in TRUTH_SCHEMAS}
        locales = sorted({loc for _, _, cs in REGIONS.values() for _, loc, _ in cs})
        self.fakers = {loc: Faker(loc) for loc in locales}
        self.rng = None
        self.day = None

    # ---------- setup / persistence ----------
    @classmethod
    def new(cls, start):
        sim = cls({"seed": SEED, "history_start": start.isoformat(), "current_date": None,
                   "next_id": {name: 1 for name in RAW_SCHEMAS}, "pending_hires": []})
        sim._build_indexes()
        return sim

    @classmethod
    def load(cls):
        sim = cls(json.loads(STATE_FILE.read_text()))
        for name in RAW_SCHEMAS:
            sim.rows[name] = read_parquet(RAW_DIR / f"{name}.parquet")
        for name in TRUTH_SCHEMAS:
            key = ID_COLUMN[name]
            sim.truth[name] = {r[key]: r for r in read_parquet(STATE_DIR / f"{name}.parquet")}
        sim._build_indexes()
        return sim

    def save(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        for name, schema in RAW_SCHEMAS.items():
            write_parquet(RAW_DIR / f"{name}.parquet", self.rows[name], schema)
        for name, schema in TRUTH_SCHEMAS.items():
            write_parquet(STATE_DIR / f"{name}.parquet", list(self.truth[name].values()), schema)
        STATE_FILE.write_text(json.dumps(self.state, indent=2))
        # Like a real CRM export: states which day the data is complete through.
        (RAW_DIR / "export_info.json").write_text(
            json.dumps({"data_through": self.state["current_date"]}, indent=2))

    def _build_indexes(self):
        self.reps = {r["rep_id"]: r for r in self.rows["sales_reps"]}
        self.accounts = {a["account_id"]: a for a in self.rows["accounts"]}
        self.domains = {a["domain"] for a in self.rows["accounts"]}
        self.accounts_by_region = {region: [] for region in REGIONS}
        for a in self.rows["accounts"]:
            self.accounts_by_region[a["region"]].append(a["account_id"])
        self.contacts_by_account = {}
        for c in self.rows["contacts"]:
            self.contacts_by_account.setdefault(c["account_id"], []).append(c["contact_id"])
        self.open_leads = {l["lead_id"]: l for l in self.rows["leads"] if l["status"] in ("New", "Working")}
        self.deals = {d["deal_id"]: d for d in self.rows["deals"]}
        self.open_deals = {d["deal_id"]: d for d in self.rows["deals"] if not d["is_closed"]}
        self.open_deal_count = {}
        for d in self.open_deals.values():
            self.open_deal_count[d["account_id"]] = self.open_deal_count.get(d["account_id"], 0) + 1
        self.customer_since = {}
        for d in self.rows["deals"]:
            if d["is_won"]:
                won = d["closed_at"].date()
                if d["account_id"] not in self.customer_since or won < self.customer_since[d["account_id"]]:
                    self.customer_since[d["account_id"]] = won

    def _next_id(self, table):
        new_id = self.state["next_id"][table]
        self.state["next_id"][table] += 1
        return new_id

    def _seed_day(self, day_number):
        # Every simulated day gets its own random stream -> reproducible, and a
        # day simulated later from saved state gets the same numbers.
        self.rng = np.random.default_rng([SEED, day_number])
        for i, f in enumerate(self.fakers.values()):
            f.seed_instance(SEED * 1_000_003 + day_number * 97 + i)

    # ---------- small helpers ----------
    def _ts(self, day=None):
        """A random business-hours timestamp on the given day."""
        day = day or self.day
        return datetime.combine(day, time(8, 0)) + timedelta(minutes=int(self.rng.integers(0, 600)))

    def _choice(self, options, weights):
        return options[int(self.rng.choice(len(options), p=np.array(weights) / sum(weights)))]

    def _pick_country(self, region):
        countries = REGIONS[region][2]
        country, locale, _ = self._choice(countries, [w for _, _, w in countries])
        return country, locale

    def _pick_rep(self, region, exclude=None):
        active = [rid for rid, r in self.reps.items() if r["is_active"] and rid != exclude]
        in_region = [rid for rid in active if self.reps[rid]["region"] == region]
        pool = in_region or active
        return pool[int(self.rng.integers(len(pool)))]

    def _log(self, deal, change_type, from_stage=None, at=None):
        self.rows["deal_stage_history"].append({
            "history_id": self._next_id("deal_stage_history"), "deal_id": deal["deal_id"],
            "changed_at": at or self._ts(), "change_type": change_type,
            "from_stage": from_stage, "to_stage": deal["stage"],
            "amount_usd": deal["amount_usd"], "probability_pct": deal["probability_pct"],
            "expected_close_date": deal["expected_close_date"], "owner_rep_id": deal["owner_rep_id"],
        })

    def _activity(self, kind, direction, outcome, rep_id, contact_id, account_id,
                  lead_id=None, deal_id=None):
        duration = None
        if kind == "call" and outcome == "connected":
            duration = int(self.rng.integers(4, 41))
        elif kind == "meeting" and outcome == "held":
            duration = int(self._choice([30, 45, 60], [0.5, 0.2, 0.3]))
        self.rows["activities"].append({
            "activity_id": self._next_id("activities"), "activity_at": self._ts(),
            "activity_type": kind, "direction": direction, "outcome": outcome,
            "duration_min": duration, "rep_id": rep_id, "contact_id": contact_id,
            "account_id": account_id, "lead_id": lead_id, "deal_id": deal_id,
        })

    # ---------- entity creation ----------
    def _new_rep(self, region, hire_date):
        _, locale = self._pick_country(region)
        f = self.fakers[locale]
        first, last = f.first_name(), f.last_name()
        rep_id = self._next_id("sales_reps")
        rep = {"rep_id": rep_id, "name": f"{first} {last}",
               "email": f"{slug(first)}.{slug(last)}@kairo.example", "region": region,
               "hire_date": hire_date, "termination_date": None, "is_active": True,
               "annual_quota_usd": float(round(self.rng.uniform(550_000, 950_000), -4))}
        self.rows["sales_reps"].append(rep)
        self.reps[rep_id] = rep
        self.truth["reps_truth"][rep_id] = {
            "rep_id": rep_id,
            "skill": float(self.rng.normal(0, 0.45)),        # adds to win log-odds
            "optimism": float(self.rng.beta(2, 3)),          # 0 = realistic, 1 = very rosy
            "activity_level": float(self.rng.lognormal(0, 0.25)),
        }
        return rep

    def _new_account(self, region, created_at):
        rng = self.rng
        country, locale = self._pick_country(region)
        name = self.fakers["en_US"].company()
        domain = f"{slug(name)[:24]}.com"
        while domain in self.domains:
            domain = f"{slug(name)[:20]}{int(rng.integers(10, 999))}.com"
        self.domains.add(domain)
        industry = self._choice(list(INDUSTRIES), [v[0] for v in INDUSTRIES.values()])
        size = self._choice(list(SIZE_BANDS), [v[0] for v in SIZE_BANDS.values()])
        lo, hi = SIZE_BANDS[size][3]
        employees = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        account_id = self._next_id("accounts")
        missing_size = rng.random() < 0.07
        account = {
            "account_id": account_id, "account_name": name, "domain": domain,
            "industry": None if rng.random() < 0.04 else industry,
            "employee_count": None if missing_size else employees,
            "size_band": None if missing_size else size,
            "region": region, "country": country,
            "annual_revenue_usd": None if rng.random() < 0.15 else float(round(employees * rng.uniform(80_000, 250_000), -5)),
            "owner_rep_id": self._pick_rep(region), "created_at": created_at,
        }
        self.rows["accounts"].append(account)
        self.accounts[account_id] = account
        self.accounts_by_region[region].append(account_id)
        self.truth["accounts_truth"][account_id] = {
            "account_id": account_id, "true_industry": industry,
            "true_size_band": size, "name_locale": locale}
        return account

    def _new_contact(self, account, created_at):
        rng = self.rng
        f = self.fakers[self.truth["accounts_truth"][account["account_id"]]["name_locale"]]
        seniority = self._choice(list(SENIORITY), [v[0] for v in SENIORITY.values()])
        titles = SENIORITY[seniority][2]
        title = titles[int(rng.integers(len(titles)))]
        first, last = f.first_name(), f.last_name()
        has_title = rng.random() > 0.06
        contact = {
            "contact_id": self._next_id("contacts"), "account_id": account["account_id"],
            "first_name": first, "last_name": last,
            "email": f"{slug(first)}.{slug(last)}@{account['domain']}",
            "phone": f.phone_number() if rng.random() > 0.22 else None,
            "job_title": title if has_title else None,
            "seniority": seniority if has_title else None,
            "created_at": created_at,
        }
        self._store_contact(contact, seniority)
        if rng.random() < 0.025:  # someone re-enters the same person by hand
            dup = dict(contact, contact_id=self._next_id("contacts"),
                       created_at=created_at + timedelta(minutes=int(rng.integers(5, 120))), phone=None)
            variant = int(rng.integers(3))
            if variant == 0:
                dup["email"] = dup["email"].upper()
            elif variant == 1:
                dup["email"] = f"{slug(first)}{slug(last)}{int(rng.integers(1, 99))}@gmail.com"
            else:
                dup["first_name"], dup["job_title"], dup["seniority"] = first.lower(), None, None
            self._store_contact(dup, seniority)
        return contact

    def _store_contact(self, contact, true_seniority):
        self.rows["contacts"].append(contact)
        self.contacts_by_account.setdefault(contact["account_id"], []).append(contact["contact_id"])
        self.truth["contacts_truth"][contact["contact_id"]] = {
            "contact_id": contact["contact_id"], "true_seniority": true_seniority}

    def _new_lead(self):
        rng, day = self.rng, self.day
        created_at = self._ts()
        region = self._choice(list(REGIONS), [v[0] for v in REGIONS.values()])
        source = self._choice(list(SOURCES), [v[0] for v in SOURCES.values()])
        existing = self.accounts_by_region[region]
        if existing and rng.random() < 0.25:
            account = self.accounts[existing[int(rng.integers(len(existing)))]]
            known = self.contacts_by_account.get(account["account_id"], [])
            if known and rng.random() < 0.3:
                contact_id = known[int(rng.integers(len(known)))]
            else:
                contact_id = self._new_contact(account, created_at)["contact_id"]
        else:
            account = self._new_account(region, created_at)
            contact_id = self._new_contact(account, created_at)["contact_id"]

        acct_truth = self.truth["accounts_truth"][account["account_id"]]
        seniority = self.truth["contacts_truth"][contact_id]["true_seniority"]
        logit = (LEAD_BASE + INDUSTRIES[acct_truth["true_industry"]][1]
                 + SIZE_BANDS[acct_truth["true_size_band"]][1] + SOURCES[source][1]
                 + SENIORITY[seniority][1] + rng.normal(0, 0.5))
        p = sigmoid(logit)
        will_convert = bool(rng.random() < p)
        touch_lag = {"inbound": (0, 2), "referral": (0, 3), "event": (1, 6), "outbound": (0, 1)}[source]
        first_touch = day + timedelta(days=int(rng.integers(*touch_lag)))
        # ~5% of dead-end leads are simply forgotten and stay "Working" forever
        forgotten = not will_convert and rng.random() < 0.05
        resolve = None if forgotten else next_workday(first_touch + timedelta(days=int(3 + rng.gamma(2, 6))))

        lead_id = self._next_id("leads")
        lead = {"lead_id": lead_id, "created_at": created_at, "source": source,
                "contact_id": contact_id, "account_id": account["account_id"],
                "owner_rep_id": self._pick_rep(region), "status": "New",
                "status_changed_at": created_at, "disqualify_reason": None,
                "converted_deal_id": None}
        self.rows["leads"].append(lead)
        self.open_leads[lead_id] = lead
        self.truth["leads_truth"][lead_id] = {
            "lead_id": lead_id, "true_convert_prob": p, "will_convert": will_convert,
            "first_touch_date": first_touch, "resolve_date": resolve, "replies": 0}

    def _new_deal(self, account, contact_id, owner_id, deal_type, source, lead_id, replies, at):
        rng = self.rng
        acct_truth = self.truth["accounts_truth"][account["account_id"]]
        industry, size = acct_truth["true_industry"], acct_truth["true_size_band"]
        rep_truth = self.truth["reps_truth"][owner_id]
        optimism = rep_truth["optimism"]

        tier = self._choice(list(PRODUCT_TIERS), TIER_MIX[size])
        true_amount = PRODUCT_TIERS[tier] * SIZE_BANDS[size][4] * rng.lognormal(0, 0.35)
        if deal_type == "Expansion":
            true_amount *= 0.4
        true_amount = float(round(true_amount, -2))

        # The hidden "true" chance of winning - what the ML models try to learn.
        logit = (DEAL_BASE + INDUSTRIES[industry][1] + SIZE_BANDS[size][1]
                 + (EXPANSION_EFFECT if deal_type == "Expansion" else SOURCES[source][1])
                 + 0.30 * min(replies, 5) + rep_truth["skill"] + rng.normal(0, 0.35))
        p = sigmoid(logit)
        will_win = bool(rng.random() < p)
        final_idx = 4 if will_win else int(rng.choice(5, p=LOSS_STAGE_WEIGHTS))
        cycle_mult = SIZE_BANDS[size][2] * INDUSTRIES[industry][2] * (0.7 if deal_type == "Expansion" else 1.0)

        # The rep's own guesses: optimists expect bigger deals that close sooner.
        planned_days = sum(STAGE_DAYS) * cycle_mult * (1 - 0.4 * optimism) * rng.uniform(0.8, 1.1)
        entered_amount = None if rng.random() < 0.10 else float(round(true_amount * (1 + 0.3 * optimism * rng.random()), -2))

        deal_id = self._next_id("deals")
        deal = {
            "deal_id": deal_id,
            "deal_name": f"{account['account_name']} - {tier}" + (" (Expansion)" if deal_type == "Expansion" else ""),
            "account_id": account["account_id"], "primary_contact_id": contact_id,
            "lead_id": lead_id, "owner_rep_id": owner_id, "deal_type": deal_type,
            "source": None if (source and rng.random() < 0.03) else source,
            "product_tier": tier, "stage": STAGES[0], "amount_usd": entered_amount,
            "probability_pct": self._rep_probability(0, optimism),
            "expected_close_date": next_workday(self.day + timedelta(days=int(planned_days))),
            "created_at": at, "stage_changed_at": at, "closed_at": None,
            "is_closed": False, "is_won": False, "lost_reason": None,
        }
        self.rows["deals"].append(deal)
        self.deals[deal_id] = deal
        self.open_deals[deal_id] = deal
        self.open_deal_count[account["account_id"]] = self.open_deal_count.get(account["account_id"], 0) + 1
        self.truth["deals_truth"][deal_id] = {
            "deal_id": deal_id, "true_win_prob": p, "will_win": will_win,
            "final_stage": STAGES[final_idx], "true_amount": true_amount,
            "discount": float(rng.uniform(0, 0.2)), "cycle_mult": cycle_mult,
            "engagement_level": float(rng.lognormal(0, ENGAGEMENT_SPREAD)),
            "next_event_date": None, "dark_since": None}
        self._schedule(deal)
        self._log(deal, "created", at=at)
        return deal

    @staticmethod
    def _rep_probability(stage_idx, optimism):
        return int(min(95, STAGE_PROB[stage_idx] + 5 * round(optimism * 4)))

    def _schedule(self, deal):
        t = self.truth["deals_truth"][deal["deal_id"]]
        mean = STAGE_DAYS[STAGES.index(deal["stage"])] * t["cycle_mult"] * self._signal(t, "stage_speed")
        t["next_event_date"] = next_workday(self.day + timedelta(days=max(1, int(round(self.rng.gamma(2.0, mean / 2))))))

    # ---------- one simulated day ----------
    def simulate_day(self, day):
        self.day = day
        self._seed_day(day.toordinal())
        self._rep_changes()
        self._generate_leads()
        self._work_leads()
        self._expansion_deals()
        self._work_deals()
        self.state["current_date"] = day.isoformat()

    def _rep_changes(self):
        rng, day = self.rng, self.day
        for hire in list(self.state["pending_hires"]):
            if date.fromisoformat(hire["hire_date"]) <= day:
                self._new_rep(hire["region"], day)
                self.state["pending_hires"].remove(hire)
        for rep_id in sorted(self.reps):
            rep = self.reps[rep_id]
            same_region = sum(r["is_active"] and r["region"] == rep["region"] for r in self.reps.values())
            if rep["is_active"] and same_region > 1 and rng.random() < 1 / (365 * 4):
                rep["is_active"], rep["termination_date"] = False, day
                for deal in self.open_deals.values():
                    if deal["owner_rep_id"] == rep_id:
                        deal["owner_rep_id"] = self._pick_rep(rep["region"], exclude=rep_id)
                        self._log(deal, "owner_change", from_stage=deal["stage"])
                for lead in self.open_leads.values():
                    if lead["owner_rep_id"] == rep_id:
                        lead["owner_rep_id"] = self._pick_rep(rep["region"], exclude=rep_id)
                self.state["pending_hires"].append({
                    "region": rep["region"],
                    "hire_date": (day + timedelta(days=int(rng.integers(20, 61)))).isoformat()})

    def _generate_leads(self):
        start = date.fromisoformat(self.state["history_start"])
        growth = 1 + 0.02 * (self.day - start).days / 30.4
        weekday = 1.0 if is_workday(self.day) else 0.12
        expected = LEADS_PER_WORKDAY * growth * weekday * MONTH_FACTOR.get(self.day.month, 1.0)
        for _ in range(int(self.rng.poisson(expected))):
            self._new_lead()

    def _work_leads(self):
        rng, day = self.rng, self.day
        for lead in list(self.open_leads.values()):
            t = self.truth["leads_truth"][lead["lead_id"]]
            if lead["status"] == "New":
                if day < t["first_touch_date"]:
                    continue
                lead["status"], lead["status_changed_at"] = "Working", self._ts()

            rep_truth = self.truth["reps_truth"][lead["owner_rep_id"]]
            ids = dict(rep_id=lead["owner_rep_id"], contact_id=lead["contact_id"],
                       account_id=lead["account_id"], lead_id=lead["lead_id"])
            age = (day - lead["created_at"].date()).days
            if is_workday(day) and age <= 30 and rng.random() < 0.45 * rep_truth["activity_level"]:
                kind = "email" if rng.random() < 0.65 else "call"
                reply_p = 0.05 + 0.30 * t["true_convert_prob"] + (0.15 if t["will_convert"] else 0.0)
                replied = rng.random() < reply_p
                if kind == "email":
                    self._activity("email", "outbound", "sent", **ids)
                    if replied:
                        self._activity("email", "inbound", "replied", **ids)
                else:
                    self._activity("call", "outbound", "connected" if replied else
                                   self._choice(["voicemail", "no answer"], [0.5, 0.5]), **ids)
                if replied:
                    t["replies"] += 1
                    if rng.random() < 0.35:
                        self._activity("meeting", "outbound", "held", **ids)
                        t["replies"] += 1

            if t["resolve_date"] is not None and day >= t["resolve_date"]:
                self._resolve_lead(lead, t)

    def _resolve_lead(self, lead, t):
        rng = self.rng
        now = self._ts()
        lead["status_changed_at"] = now
        del self.open_leads[lead["lead_id"]]
        if t["will_convert"]:
            lead["status"] = "Converted"
            deal = self._new_deal(self.accounts[lead["account_id"]], lead["contact_id"],
                                  lead["owner_rep_id"], "New Business", lead["source"],
                                  lead["lead_id"], t["replies"], now)
            lead["converted_deal_id"] = deal["deal_id"]
        elif rng.random() < 0.65:
            lead["status"] = "Disqualified"
            if rng.random() > 0.2:
                lead["disqualify_reason"] = DISQUALIFY_REASONS[int(rng.integers(len(DISQUALIFY_REASONS)))]
        else:
            lead["status"] = "Nurture"

    def _expansion_deals(self):
        for account_id in sorted(self.customer_since):
            if self.open_deal_count.get(account_id, 0) or (self.day - self.customer_since[account_id]).days < 90:
                continue
            if self.rng.random() < 1 / 300:
                account = self.accounts[account_id]
                contacts = self.contacts_by_account[account_id]
                owner = account["owner_rep_id"]
                if not self.reps[owner]["is_active"]:
                    owner = self._pick_rep(account["region"])
                self._new_deal(account, contacts[int(self.rng.integers(len(contacts)))], owner,
                               "Expansion", None, None, 0, self._ts())

    def _work_deals(self):
        for deal in list(self.open_deals.values()):
            t = self.truth["deals_truth"][deal["deal_id"]]
            self._deal_activities(deal, t)
            if self.day >= t["next_event_date"]:
                self._deal_event(deal, t)
            if not deal["is_closed"]:
                self._maybe_slip_close_date(deal)

    def _deal_activities(self, deal, t):
        rng = self.rng
        activity_level = self.truth["reps_truth"][deal["owner_rep_id"]]["activity_level"]
        workday_factor = 1.0 if is_workday(self.day) else 0.08
        ids = dict(rep_id=deal["owner_rep_id"], account_id=deal["account_id"], deal_id=deal["deal_id"])
        contacts = self.contacts_by_account[deal["account_id"]]

        def contact():
            if rng.random() < 0.65:
                return deal["primary_contact_id"]
            return contacts[int(rng.integers(len(contacts)))]

        if t["dark_since"] is not None:
            # The buyer has gone quiet: the rep keeps chasing, nobody answers.
            for _ in range(int(rng.poisson(0.10 * activity_level * workday_factor))):
                if rng.random() < 0.7:
                    self._activity("email", "outbound", "no reply", contact_id=contact(), **ids)
                else:
                    self._activity("call", "outbound", "no answer", contact_id=contact(), **ids)
            return

        i = STAGES.index(deal["stage"])
        engagement = t["engagement_level"]
        rate = ACT_RATE[i] * activity_level * workday_factor * engagement * self._signal(t, "activity")
        if not t["will_win"] and STAGES[i] == t["final_stage"]:
            rate *= LOSING_COOL_OFF  # losing deals cool off before they die
        for _ in range(int(rng.poisson(rate))):
            kind = self._choice(["email", "call", "meeting"], ACT_MIX[i])
            if kind == "email":
                inbound_share = min(0.7, self._signal(t, "inbound_email") * math.sqrt(engagement))
                inbound = rng.random() < inbound_share
                self._activity("email", "inbound" if inbound else "outbound",
                               "replied" if inbound else "sent", contact_id=contact(), **ids)
            elif kind == "call":
                connect = self._signal(t, "call_connect")
                outcome = self._choice(["connected", "voicemail", "no answer"],
                                       [connect, (1 - connect) / 2, (1 - connect) / 2])
                self._activity("call", "outbound", outcome, contact_id=contact(), **ids)
            else:
                held = self._signal(t, "meeting_held")
                self._activity("meeting", "outbound", "held" if rng.random() < held else "no-show",
                               contact_id=contact(), **ids)

    @staticmethod
    def _signal(t, name):
        if_won, if_lost = WIN_SIGNAL[name]
        return if_won if t["will_win"] else if_lost

    def _deal_event(self, deal, t):
        rng = self.rng
        i = STAGES.index(deal["stage"])
        optimism = self.truth["reps_truth"][deal["owner_rep_id"]]["optimism"]
        if t["dark_since"] is not None:
            reason = "No decision / went dark" if rng.random() > 0.3 else None
            self._close(deal, won=False, reason=reason)
        elif t["will_win"] and i == 4:
            self._close(deal, won=True)
        elif STAGES[i] != t["final_stage"] or t["will_win"]:
            self._advance(deal, t, i + 1, optimism)
        elif rng.random() < 0.5:
            reason = LOST_REASONS[int(rng.integers(len(LOST_REASONS)))] if rng.random() > 0.25 else None
            self._close(deal, won=False, reason=reason)
        else:
            # Goes dark. The rep only admits it's lost weeks (or months) later;
            # optimistic reps keep dead deals in the pipeline longer.
            t["dark_since"] = self.day
            lag = 14 + rng.gamma(2, 15 * (1 + 2 * optimism))
            if rng.random() < 0.10:
                lag += 150
            t["next_event_date"] = next_workday(self.day + timedelta(days=int(lag)))

    def _advance(self, deal, t, new_idx, optimism):
        rng = self.rng
        old = deal["stage"]
        deal["stage"] = STAGES[new_idx]
        deal["stage_changed_at"] = self._ts()
        deal["probability_pct"] = self._rep_probability(new_idx, optimism)
        if deal["stage"] == "Proposal":      # formal quote
            deal["amount_usd"] = float(round(t["true_amount"] * (1 + 0.2 * optimism * rng.random()), -2))
        elif deal["stage"] == "Negotiation":  # discounting starts
            deal["amount_usd"] = float(round(t["true_amount"] * (1 - t["discount"]), -2))
        elif deal["amount_usd"] is None and rng.random() < 0.5:
            deal["amount_usd"] = float(round(t["true_amount"] * (1 + 0.3 * optimism * rng.random()), -2))
        self._log(deal, "stage_change", from_stage=old, at=deal["stage_changed_at"])
        self._schedule(deal)

    def _close(self, deal, won, reason=None):
        old = deal["stage"]
        now = self._ts()
        t = self.truth["deals_truth"][deal["deal_id"]]
        deal.update(stage=WON if won else LOST, is_closed=True, is_won=won, closed_at=now,
                    stage_changed_at=now, expected_close_date=self.day,
                    probability_pct=100 if won else 0, lost_reason=reason)
        if won:
            deal["amount_usd"] = float(round(t["true_amount"] * (1 - t["discount"]), -2))
            self.customer_since.setdefault(deal["account_id"], self.day)
        t["next_event_date"] = None
        del self.open_deals[deal["deal_id"]]
        self.open_deal_count[deal["account_id"]] -= 1
        self._log(deal, "stage_change", from_stage=old, at=now)

    def _maybe_slip_close_date(self, deal):
        # Close date has passed and the deal is still open: sooner or later the
        # rep pushes it out. Optimists push it by less, so it slips again.
        if deal["expected_close_date"] >= self.day or not is_workday(self.day):
            return
        optimism = self.truth["reps_truth"][deal["owner_rep_id"]]["optimism"]
        if self.rng.random() < 0.12:
            push = int(self.rng.integers(10, 46) * (1 - 0.5 * optimism))
            deal["expected_close_date"] = next_workday(self.day + timedelta(days=max(5, push)))
            self._log(deal, "close_date_change", from_stage=deal["stage"])

    # ---------- first run ----------
    def bootstrap(self, start):
        """Company switched to this CRM on `start`: reps + an imported prospect list."""
        self._seed_day(0)
        self.day = start
        for region, (_, n_reps, _) in REGIONS.items():
            for _ in range(n_reps):
                self._new_rep(region, start - timedelta(days=int(self.rng.integers(60, 1800))))
        imported_at = datetime.combine(start, time(7, 0))
        for _ in range(350):
            region = self._choice(list(REGIONS), [v[0] for v in REGIONS.values()])
            account = self._new_account(region, imported_at)
            for _ in range(int(self.rng.integers(1, 4))):
                self._new_contact(account, imported_at)


# ---------------------------------------------------------------------------
def print_summary(sim, before, title):
    print(f"\n{title}")
    print(f"{'table':<22}{'rows':>9}{'new':>9}")
    for name in RAW_SCHEMAS:
        n = len(sim.rows[name])
        print(f"{name:<22}{n:>9,}{n - before.get(name, 0):>9,}")
    deals = sim.rows["deals"]
    won = sum(d["is_won"] for d in deals)
    lost = sum(d["is_closed"] and not d["is_won"] for d in deals)
    dark = sum(t["dark_since"] is not None for t in sim.truth["deals_truth"].values()
               if t["deal_id"] in sim.open_deals)
    overdue = sum(d["expected_close_date"] < sim.day for d in sim.open_deals.values())
    print(f"\nSimulated date: {sim.state['current_date']}  (history started {sim.state['history_start']})")
    print(f"Deals: {won:,} won, {lost:,} lost, {len(sim.open_deals):,} open "
          f"(win rate {won / max(won + lost, 1):.0%}); {dark} open deals have secretly gone quiet, "
          f"{overdue} open deals have a close date in the past")


def main():
    parser = argparse.ArgumentParser(description="Simulate Kairo's CRM data.")
    parser.add_argument("--reset", action="store_true", help="delete generated data and rebuild history")
    parser.add_argument("--end-date", type=date.fromisoformat, help="last day of initial history (default: yesterday)")
    parser.add_argument("--force", action="store_true", help="allow simulating a day after today")
    args = parser.parse_args()

    if args.reset:
        for folder in (RAW_DIR, STATE_DIR):
            shutil.rmtree(folder, ignore_errors=True)

    if STATE_FILE.exists():
        if args.end_date:
            print("Note: --end-date only applies to the first run (or with --reset); ignoring it.")
        sim = CRMSimulator.load()
        before = {name: len(rows) for name, rows in sim.rows.items()}
        next_day = date.fromisoformat(sim.state["current_date"]) + timedelta(days=1)
        if next_day > date.today() and not args.force:
            print(f"The CRM is already simulated up to {sim.state['current_date']}. "
                  f"Simulating {next_day} would be a day in the future, so nothing was changed. "
                  f"(Use --force to do it anyway.)")
            return
        sim.simulate_day(next_day)
        sim.save()
        print_summary(sim, before, f"Advanced the CRM by one day: {next_day}")
    else:
        end = args.end_date or date.today() - timedelta(days=1)
        start = end - timedelta(days=HISTORY_DAYS - 1)
        sim = CRMSimulator.new(start)
        sim.bootstrap(start)
        day = start
        while day <= end:
            sim.simulate_day(day)
            day += timedelta(days=1)
        sim.save()
        print_summary(sim, {}, f"Built CRM history: {start} -> {end} ({HISTORY_DAYS} days)")


if __name__ == "__main__":
    main()
