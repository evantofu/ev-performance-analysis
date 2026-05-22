"""
EV Data Collection
==================
Pulls from three free sources and writes CSVs to data/raw/ in the exact
column shapes expected by ev_analysis.py and app.py.

Sources
-------
1. fueleconomy.gov REST API   — EPA-tested MPGe, range, specs (no key needed)
2. OpenEV Data (GitHub)       — richer charging + powertrain specs (no key needed)
3. NREL AFDC API              — US charging stations (free key from developer.nlr.gov)

Usage
-----
    # Minimal — vehicles + stations, skips OpenEV if GitHub is slow
    python data_collection.py --nrel-key YOUR_KEY

    # Full
    python data_collection.py --nrel-key YOUR_KEY --include-openev

    # Dry run (no network calls, generates synthetic data for testing)
    python data_collection.py --dry-run

Get your free NREL key (instant, no approval):
    https://developer.nlr.gov/signup/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import os

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

STAMP = datetime.now().strftime("%Y%m%d")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ev-explorer-data-collector/1.0"})


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    """GET with retries and a short back-off."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            log.warning("  Retry %d/%d after %ds (%s)", attempt + 1, retries, wait, e)
            time.sleep(wait)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — fueleconomy.gov bulk CSV download
# No key, no pagination, no XML parsing.  One request fetches every vehicle
# EPA has ever tested (1984–present) in a single ~4 MB CSV.
# Download page: https://www.fueleconomy.gov/feg/download.shtml
#
# The menu-based REST API returns XML for the year/make/model menus, which
# cannot be reliably parsed as JSON.  The bulk CSV is the correct approach
# for any batch / analysis workload — faster and more complete.
# ══════════════════════════════════════════════════════════════════════════════
FE_CSV_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"


def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def collect_vehicles_fe(start_year: int = 2019) -> pd.DataFrame:
    """
    Download the full fueleconomy.gov vehicles CSV (zipped, ~4 MB) and
    return EV/PHEV rows from start_year onward.

    Column mapping keeps the same output shape used by ev_analysis.py
    and app.py so nothing downstream needs to change.
    """
    log.info("── fueleconomy.gov: downloading bulk CSV (one request, ~4 MB)")
    r = _get(FE_CSV_URL)

    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            raw = pd.read_csv(f, low_memory=False)

    log.info("  Full dataset: %d rows — filtering EVs from %d+", len(raw), start_year)

    # Filter to EVs and PHEVs only
    ev_mask = (
        raw["fuelType1"].str.contains("Electricity", na=False)
        | raw["fuelType2"].str.contains("Electricity", na=False)
        | raw["atvType"].str.contains("EV|PHEV|Plug-in", na=False, case=False)
    )
    df = raw[ev_mask & (raw["year"] >= start_year)].copy()
    log.info("  EV rows after filter: %d", len(df))

    # Rename to the column names the rest of the pipeline expects
    col_map = {
        "id":            "fueleconomy_id",
        "year":          "year",
        "make":          "make",
        "model":         "model",
        "trany":         "trany",
        "drive":         "drive",
        "VClass":        "VClass",
        "comb08U":       "combined_mpge",   # MPGe unrounded (falls back below)
        "comb08":        "_comb08",
        "city08U":       "city_mpge",
        "city08":        "_city08",
        "highway08U":    "highway_mpge",
        "highway08":     "_highway08",
        "combE":         "kwh_per_100mi",
        "range":         "range_miles",
        "cityE":         "range_city",
        "highwayE":      "range_highway",
        "co2TailpipeAGpm": "co2_tailpipe_gpm",
        "ghgScore":      "ghg_score",
        "smartwayScore": "smog_score",
        "fuelCost08":    "annual_fuel_cost_usd",
        "fuelType1":     "fuel_type",
        "fuelType2":     "fuel_type2",
        "phevBlended":   "is_phev",
        "evMotor":       "ev_motor",
    }
    present = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=present)

    # Prefer unrounded MPGe cols; fall back to rounded when unrounded is zero/NaN
    for unr, fallback, out in [
        ("combined_mpge", "_comb08",    "combined_mpge"),
        ("city_mpge",     "_city08",    "city_mpge"),
        ("highway_mpge",  "_highway08", "highway_mpge"),
    ]:
        if unr in df.columns and fallback in df.columns:
            df[out] = df[unr].where(df[unr].notna() & (df[unr] != 0), df[fallback])
            df.drop(columns=[c for c in [fallback] if c != out and c in df.columns],
                    inplace=True, errors="ignore")

    # Columns enriched later from OpenEV — initialise as NaN
    for col in ["battery_capacity_kwh", "msrp_base", "charge_240v_hrs",
                "fast_charge_minutes", "max_ac_kw", "max_dc_kw",
                "connector_type", "acceleration_0_60"]:
        if col not in df.columns:
            df[col] = None

    df["is_phev"] = df.get("is_phev", pd.Series(False, index=df.index)).astype(bool)
    df = df.reset_index(drop=True)
    log.info("  Returning %d EV rows", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — OpenEV Data (GitHub release, no key)
# Community-maintained EV specs with charging details not in EPA data.
# Docs: https://open-ev-data.github.io/
# ══════════════════════════════════════════════════════════════════════════════
OPENEV_RELEASE_API = (
    "https://api.github.com/repos/open-ev-data/open-ev-data-dataset/releases/latest"
)


def _find_openev_json_url() -> str:
    """Fetch the latest GitHub release and return the JSON asset download URL."""
    log.info("── OpenEV Data: resolving latest release")
    meta = _get(OPENEV_RELEASE_API).json()
    assets = meta.get("assets", [])
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".json") and "open-ev-data" in name.lower():
            url = asset["browser_download_url"]
            log.info("  Found asset: %s", name)
            return url
    raise RuntimeError(
        "Could not find JSON asset in latest OpenEV release. "
        f"Assets found: {[a['name'] for a in assets]}"
    )


def _first(*values):
    """Return first non-None, non-empty value."""
    for v in values:
        if v is not None and v != "" and v != []:
            return v
    return None


def _obj_name(field) -> str | None:
    """
    OpenEV v1.24+ stores make/model as {'slug': 'tesla', 'name': 'Tesla'}.
    Also handles plain strings for backwards compatibility.
    """
    if field is None:
        return None
    if isinstance(field, dict):
        return field.get("name") or field.get("slug") or None
    return str(field).strip() or None


def _extract_openev_row(v: dict) -> dict:
    """
    Extract fields from one OpenEV v1.24 record.

    Confirmed schema from live data:
      make/model        → {'slug': ..., 'name': ...}
      battery           → {'pack_capacity_kwh_net': ..., 'pack_capacity_kwh_gross': ...}
      charging.ac       → {'max_power_kw': ...}
      charging.dc       → {'max_power_kw': ...}
      range.rated       → [{'cycle': 'wltp'|'epa', 'range_km': ...}]
      charge_ports      → [{'connector': 'ccs2'|'nacs'|..., 'kind': ...}]
      performance       → {'acceleration_0_60_mph_s': ...}
      msrp              → not present in this dataset
    """
    KM_TO_MI = 0.621371

    # ── Make / model ──────────────────────────────────────────────────────────
    make  = _obj_name(v.get("make"))
    model = _obj_name(v.get("model"))
    year  = v.get("year")

    # ── Battery ───────────────────────────────────────────────────────────────
    bat = v.get("battery") or {}
    batt_kwh = _float(_first(
        bat.get("pack_capacity_kwh_net"),
        bat.get("pack_capacity_kwh_gross"),
        bat.get("usable_kwh"), bat.get("capacity_kwh"),
    ))

    # ── Range — rated list, prefer EPA cycle then WLTP ───────────────────────
    range_mi = None
    rated = (v.get("range") or {}).get("rated") or []
    if isinstance(rated, list):
        # Try EPA first
        for entry in rated:
            if isinstance(entry, dict) and entry.get("cycle") == "epa":
                range_mi = _float(entry.get("range_km", 0)) * KM_TO_MI if entry.get("range_km") else None
                if range_mi: break
        # Fall back to WLTP
        if range_mi is None:
            for entry in rated:
                if isinstance(entry, dict) and entry.get("range_km"):
                    range_mi = _float(entry["range_km"]) * KM_TO_MI
                    break
    if range_mi:
        range_mi = round(range_mi, 0)

    # ── Charging ──────────────────────────────────────────────────────────────
    chg    = v.get("charging") or {}
    ac     = chg.get("ac") or {}
    dc     = chg.get("dc") or {}
    max_ac = _float(_first(ac.get("max_power_kw"), ac.get("max_kw")))
    max_dc = _float(_first(dc.get("max_power_kw"), dc.get("max_kw")))

    # DC fast charge time to 80% — not in this dataset, estimate if possible
    dc_min = _float(dc.get("time_to_80_min"))

    # Level 2 time to full — derive from battery/charger if not explicit
    charge_hrs = _float(ac.get("time_to_full_h"))
    if charge_hrs is None and batt_kwh and max_ac and max_ac > 0:
        charge_hrs = round(batt_kwh / max_ac, 1)

    # ── Connector — charge_ports list ─────────────────────────────────────────
    connector = None
    ports = v.get("charge_ports") or []
    if isinstance(ports, list) and ports:
        port = ports[0]
        if isinstance(port, dict):
            raw_conn = _first(port.get("connector"), port.get("standard"),
                              port.get("type"), port.get("connector_type"))
            if raw_conn:
                # Normalise to friendly names
                conn_map = {
                    "ccs1": "CCS", "ccs2": "CCS2", "nacs": "NACS",
                    "type2": "Type 2", "chademo": "CHAdeMO",
                    "j1772": "J1772", "gbt": "GB/T",
                }
                connector = conn_map.get(str(raw_conn).lower(), str(raw_conn).upper())

    # ── Performance ───────────────────────────────────────────────────────────
    perf  = v.get("performance") or {}
    accel = _float(_first(
        perf.get("acceleration_0_60_mph_s"),
        perf.get("zero_to_60_mph_s"),
        perf.get("accel_0_60"),
    ))

    # ── Price — not in OpenEV v1.24, leave as None ───────────────────────────
    msrp = _float(_first(
        v.get("msrp_usd"), v.get("base_price_usd"),
        v.get("price_usd"), v.get("msrp"), v.get("price"),
    ))

    return {
        "make":                 make,
        "model":                model,
        "year":                 year,
        "battery_capacity_kwh": batt_kwh,
        "range_miles":          range_mi,
        "charge_240v_hrs":      charge_hrs,
        "fast_charge_minutes":  dc_min,
        "max_ac_kw":            max_ac,
        "max_dc_kw":            max_dc,
        "connector_type":       connector,
        "acceleration_0_60":    accel,
        "msrp_base":            msrp,
    }


def collect_openev() -> pd.DataFrame:
    """Download and normalise the OpenEV JSON dataset."""
    url = _find_openev_json_url()
    log.info("  Downloading OpenEV dataset…")
    raw = _get(url).json()

    # Unwrap if top-level is a dict container
    if isinstance(raw, dict):
        for key in ("vehicles", "data", "cars", "evs", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                log.info("  Unwrapped JSON key '%s'", key)
                break
        else:
            # Last resort: find any list value
            for v in raw.values():
                if isinstance(v, list) and len(v) > 0:
                    raw = v
                    break

    if not raw:
        log.warning("  OpenEV: empty dataset after unwrap")
        return pd.DataFrame()

    # Log first record keys to help diagnose future schema changes
    log.info("  OpenEV first record keys: %s", list(raw[0].keys())[:15])


    rows = [_extract_openev_row(v) for v in raw]
    df = pd.DataFrame(rows)

    # Report how many fields were actually populated
    for col in ["make", "model", "year", "battery_capacity_kwh", "msrp_base",
                "fast_charge_minutes", "max_dc_kw", "acceleration_0_60"]:
        if col in df.columns:
            filled = df[col].notna().sum()
            log.info("  OpenEV %-25s %d/%d filled", col, filled, len(df))

    log.info("  Collected %d vehicles from OpenEV Data", len(df))
    return df


# Make name aliases — EPA name → OpenEV name (or vice versa)
_MAKE_ALIASES = {
    "vw": "volkswagen",
    "chevy": "chevrolet",
    "mercedes": "mercedes-benz",
    "mercedes benz": "mercedes-benz",
    "bmw": "bmw",
    "land rover": "land rover",
    "rolls royce": "rolls-royce",
    "rolls-royce": "rolls-royce",
    "alfa romeo": "alfa romeo",
}


def _normalise_make(s: str) -> str:
    s = str(s).strip().lower()
    return _MAKE_ALIASES.get(s, s)


def _normalise_model(s: str) -> str:
    """Aggressively normalise model name for fuzzy join."""
    import re
    s = str(s).strip().lower()
    # Remove common suffixes that differ between datasets
    s = re.sub(r"\b(ev|electric|bev|phev|plug.in|hybrid|quattro|awd|rwd|fwd|4wd|4x4)\b", "", s)
    # Remove trim indicators
    s = re.sub(r"\b(base|standard|long range|performance|sport|plus|pro|max|ultra|limited|premium)\b", "", s)
    # Collapse whitespace and hyphens
    s = re.sub(r"[\s\-]+", " ", s).strip()
    return s


def _merge_openev(vehicles: pd.DataFrame, openev: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join OpenEV into the EPA dataset.
    Uses a three-pass strategy:
      1. Exact join on (make, model, year)
      2. Exact join on (make, model) — any year (handles model-year mismatches)
      3. Fuzzy join on (make, normalised_model) — handles trim-name differences
    """
    log.info("── Merging OpenEV enrichment into EPA data")

    enrich_cols = [c for c in [
        "battery_capacity_kwh", "charge_240v_hrs", "fast_charge_minutes",
        "max_ac_kw", "max_dc_kw", "connector_type",
        "acceleration_0_60", "msrp_base",
    ] if c in openev.columns]

    if not enrich_cols:
        log.warning("  OpenEV: no enrich columns found — skipping merge")
        return vehicles

    # ── Normalise keys ────────────────────────────────────────────────────────
    veh = vehicles.copy()
    oe  = openev.copy()

    for df in [veh, oe]:
        df["_make"]  = df["make"].apply(_normalise_make)
        df["_model"] = df["model"].astype(str).str.strip().str.lower()
        df["_model_fuzzy"] = df["_model"].apply(_normalise_model)

    veh["year"] = pd.to_numeric(veh["year"], errors="coerce")
    oe["year"]  = pd.to_numeric(oe["year"],  errors="coerce")

    oe_slim = oe[["_make", "_model", "_model_fuzzy", "year"] + enrich_cols].copy()

    # ── Pass 1: exact (make, model, year) ────────────────────────────────────
    oe1 = oe_slim.drop_duplicates(["_make", "_model", "year"])
    merged = veh.merge(oe1[["_make", "_model", "year"] + enrich_cols],
                       on=["_make", "_model", "year"], how="left", suffixes=("", "_oe1"))
    _fill(merged, enrich_cols, "_oe1")
    n1 = merged[enrich_cols[0]].notna().sum()
    log.info("  Pass 1 (exact make+model+year): %d rows enriched", n1)

    # ── Pass 2: (make, model) any year ───────────────────────────────────────
    oe2 = oe_slim.groupby(["_make", "_model"])[enrich_cols].first().reset_index()
    merged = merged.merge(oe2, on=["_make", "_model"], how="left", suffixes=("", "_oe2"))
    _fill(merged, enrich_cols, "_oe2")
    n2 = merged[enrich_cols[0]].notna().sum()
    log.info("  Pass 2 (make+model any year):   %d rows enriched", n2)

    # ── Pass 3: (make, fuzzy model) ──────────────────────────────────────────
    oe3 = oe_slim.groupby(["_make", "_model_fuzzy"])[enrich_cols].first().reset_index()
    merged = merged.merge(oe3, on=["_make", "_model_fuzzy"], how="left", suffixes=("", "_oe3"))
    _fill(merged, enrich_cols, "_oe3")
    n3 = merged[enrich_cols[0]].notna().sum()
    log.info("  Pass 3 (make+fuzzy model):       %d rows enriched", n3)

    # ── Clean up helper columns ───────────────────────────────────────────────
    merged.drop(columns=["_make", "_model", "_model_fuzzy"], inplace=True, errors="ignore")

    filled = merged[enrich_cols].notna().any(axis=1).sum()
    log.info("  Total enriched: %d/%d rows", filled, len(merged))

    if filled == 0:
        # Print diagnostic sample to help debug future schema changes
        log.warning("  No rows enriched — printing OpenEV sample for diagnosis:")
        log.warning("  OpenEV makes (first 10): %s", oe["_make"].unique()[:10].tolist())
        log.warning("  EPA    makes (first 10): %s", veh["_make"].unique()[:10].tolist())

    return merged


def _fill(df: pd.DataFrame, cols: list, suffix: str) -> None:
    """Fill NaN values in cols from col+suffix counterparts, then drop suffix cols."""
    for col in cols:
        oe_col = col + suffix
        if oe_col in df.columns:
            df[col] = df[col].combine_first(df[oe_col])
            df.drop(columns=[oe_col], inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — NREL AFDC Charging Stations
# Free key: https://developer.nlr.gov/signup/
# Docs: https://developer.nlr.gov/docs/transportation/alt-fuel-stations-v1/
#
# Note: NREL is migrating from developer.nrel.gov → developer.nlr.gov
# (deadline April 30 2026). This script uses the new domain.
# ══════════════════════════════════════════════════════════════════════════════
NREL_BASE = "https://developer.nlr.gov/api/alt-fuel-stations/v1"


def collect_stations(api_key: str, country: str = "US") -> pd.DataFrame:
    """
    Pull all open public EV charging stations from NREL AFDC.
    Returns one row per station in the shape expected by app.py.
    """
    log.info("── NREL AFDC: collecting EV stations (%s)", country)

    params = {
        "api_key":    api_key,
        "fuel_type":  "ELEC",
        "status":     "E",           # E = currently open
        "access":     "public",
        "country":    country,
        "limit":      "all",
    }

    r = _get(f"{NREL_BASE}.json", params=params)
    data = r.json()
    raw = data.get("fuel_stations", [])
    log.info("  Raw station records: %d", len(raw))

    rows = []
    for s in raw:
        rows.append({
            # ── identity ─────────────────────────────────────────────────────
            "station_id":       s.get("id"),
            "station_name":     s.get("station_name"),
            "street":           s.get("street_address"),
            "city":             s.get("city"),
            "state":            s.get("state"),
            "zip":              s.get("zip"),
            "country":          s.get("country"),
            # ── location ─────────────────────────────────────────────────────
            "latitude":         _float(s.get("latitude")),
            "longitude":        _float(s.get("longitude")),
            # ── network ──────────────────────────────────────────────────────
            "network":          s.get("ev_network") or "Non-Networked",
            "network_url":      s.get("ev_network_web"),
            # ── access ───────────────────────────────────────────────────────
            "access_code":      s.get("access_code"),           # public / private
            "access_hours":     s.get("access_days_time"),
            "facility_type":    s.get("facility_type"),
            # ── charger counts ───────────────────────────────────────────────
            "level1_count":     int(s.get("ev_level1_evse_num") or 0),
            "level2_count":     int(s.get("ev_level2_evse_num") or 0),
            "dc_fast_count":    int(s.get("ev_dc_fast_num") or 0),
            # ── connector detail ─────────────────────────────────────────────
            "connector_types":  s.get("ev_connector_types"),    # list or None
            # ── metadata ─────────────────────────────────────────────────────
            "open_date":        s.get("open_date"),
            "updated_at":       s.get("updated_at"),
        })

    df = pd.DataFrame(rows).dropna(subset=["latitude", "longitude"])
    log.info("  Stations with coordinates: %d", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic fallback (--dry-run)
# Generates realistic-looking data without any network calls, useful for
# testing the rest of the pipeline when you don't have API keys yet.
# ══════════════════════════════════════════════════════════════════════════════
def _synthetic_vehicles() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    makers = {
        "Tesla":    [("Model 3", 82, 358, 40240, 138, "NACS"),
                     ("Model Y", 82, 330, 43990, 123, "NACS"),
                     ("Model S", 100, 405, 74990, 120, "NACS")],
        "Hyundai":  [("IONIQ 6", 77, 361, 38615, 140, "CCS"),
                     ("IONIQ 5", 77, 303, 41450, 114, "CCS")],
        "Chevrolet":[("Bolt EV", 65, 259, 26500, 120, "CCS"),
                     ("Equinox EV", 85, 319, 34995, 121, "CCS")],
        "BMW":      [("i4 eDrive40", 84, 301, 52200, 107, "CCS")],
        "Ford":     [("Mustang Mach-E", 91, 312, 42995, 100, "CCS"),
                     ("F-150 Lightning", 131, 320, 49995, 78, "CCS")],
        "Rivian":   [("R1T", 135, 314, 67500, 73, "NACS"),
                     ("R1S", 135, 321, 75900, 71, "NACS")],
    }
    rows = []
    for year in range(2019, 2026):
        for make, models in makers.items():
            for model, batt, rng_mi, price, mpge, conn in models:
                noise = rng.normal(0, 0.03)
                rows.append({
                    "fueleconomy_id":       None,
                    "year":                 year,
                    "make":                 make,
                    "model":                model,
                    "combined_mpge":        round(mpge * (1 + noise), 1),
                    "city_mpge":            round(mpge * 1.03 * (1 + noise), 1),
                    "highway_mpge":         round(mpge * 0.97 * (1 + noise), 1),
                    "kwh_per_100mi":        round(100 / mpge * 33.7, 1),
                    "range_miles":          round(rng_mi * (1 + noise), 0),
                    "battery_capacity_kwh": batt,
                    "msrp_base":            price + int(year - 2019) * 500,
                    "connector_type":       conn,
                    "fuel_type":            "Electricity",
                    "is_phev":              False,
                    "trany":                "1-Speed Automatic",
                    "drive":                "Rear-Wheel Drive",
                    "VClass":               "Compact Cars",
                    "charge_240v_hrs":      round(batt / 11, 1),
                    "fast_charge_minutes":  round(batt * 0.4, 0),
                    "max_ac_kw":            11.0,
                    "max_dc_kw":            250.0 if make == "Tesla" else 150.0,
                    "acceleration_0_60":    round(rng.uniform(3.5, 6.5), 1),
                    "annual_fuel_cost_usd": round(mpge * 0.3 * 15000 / mpge, 0),
                    "co2_tailpipe_gpm":     0,
                    "ghg_score":            10,
                })
    return pd.DataFrame(rows)


def _synthetic_stations(n: int = 5000) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(7)
    networks = ["ChargePoint Network", "Tesla", "Blink Network",
                "EVgo Network", "Non-Networked", "Tesla Destination",
                "Electrify America"]
    weights  = [0.40, 0.20, 0.10, 0.08, 0.10, 0.07, 0.05]
    # Rough US bounding box
    lats = rng.uniform(25, 49, n)
    lons = rng.uniform(-124, -67, n)
    rows = []
    for i in range(n):
        network = rng.choice(networks, p=weights)
        l2 = int(rng.integers(1, 12))
        dc = int(rng.integers(0, 6)) if network != "Tesla Destination" else 0
        rows.append({
            "station_id":    i + 1,
            "station_name":  f"{network} Station #{i+1}",
            "city":          "Various",
            "state":         "CA",
            "country":       "US",
            "latitude":      round(float(lats[i]), 6),
            "longitude":     round(float(lons[i]), 6),
            "network":       network,
            "access_code":   "public",
            "level1_count":  0,
            "level2_count":  l2,
            "dc_fast_count": dc,
            "connector_types": ["CCS", "J1772"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════
def _save(df: pd.DataFrame, name: str) -> Path:
    path = RAW_DIR / f"{name}_{STAMP}.csv"
    df.to_csv(path, index=False)
    log.info("  Saved %s → %s (%d rows)", name, path, len(df))
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nrel-key",      metavar="KEY",
                   help="NREL API key (free at developer.nlr.gov/signup)")
    p.add_argument("--start-year",    type=int, default=2019,
                   help="First model year to collect (default: 2019)")
    p.add_argument("--include-openev", action="store_true",
                   help="Download and merge OpenEV Data specs")
    p.add_argument("--dry-run",       action="store_true",
                   help="Generate synthetic data only — no network calls")
    p.add_argument("--country",       default="US",
                   help="Country code for NREL station pull (default: US)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load .env so NREL_API_KEY is available as an env var.
    # CLI flag takes priority; .env is the fallback.
    load_dotenv()
    nrel_key = args.nrel_key or os.environ.get("NREL_API_KEY")
    if nrel_key and not args.nrel_key:
        log.info("  Using NREL_API_KEY from .env")

    log.info("EV Data Collection — %s", STAMP)
    log.info("Output directory: %s", RAW_DIR.resolve())

    # ── Vehicles ──────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("── DRY RUN: generating synthetic vehicle data")
        vehicles = _synthetic_vehicles()
    else:
        vehicles = collect_vehicles_fe(start_year=args.start_year)

        if args.include_openev:
            try:
                openev = collect_openev()
                vehicles = _merge_openev(vehicles, openev)
            except Exception as e:
                log.warning("OpenEV merge failed (%s) — continuing without it", e)

    _save(vehicles, "epa_vehicles")

    # ── Stations ──────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("── DRY RUN: generating synthetic station data")
        stations = _synthetic_stations()
    elif nrel_key:
        stations = collect_stations(api_key=nrel_key, country=args.country)
    else:
        log.warning(
            "No --nrel-key provided. Skipping station collection.\n"
            "  Get a free key at: https://developer.nlr.gov/signup/\n"
            "  Then rerun with: python data_collection.py --nrel-key YOUR_KEY"
        )
        stations = pd.DataFrame()

    if not stations.empty:
        _save(stations, "charging_stations")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("Collection complete.")
    log.info("  Vehicles : %d rows", len(vehicles))
    log.info("  Stations : %d rows", len(stations) if not stations.empty else 0)
    if not args.dry_run and not nrel_key:
        log.info("  (Run with --nrel-key to collect station data)")
    log.info("")
    log.info("Next step: python ev_analysis.py")


if __name__ == "__main__":
    main()