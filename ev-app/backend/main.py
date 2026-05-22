"""
EV Explorer — FastAPI Backend
==============================
Reads processed CSVs on startup, caches as JSON, serves via REST.
All heavy computation happens in ev_analysis.py — this just serves results.

Endpoints
---------
GET /api/vehicles          All consumer BEVs with cluster labels
GET /api/vehicles/{id}     Single vehicle detail
GET /api/stations          All charging stations (paginated)
GET /api/stations/nearby   Stations within radius of lat/lon
GET /api/segments          GMM cluster summaries
GET /api/trends            Manufacturer efficiency trends
GET /api/summary           Top-level KPIs for dashboard header
GET /health                Liveness probe
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()  # loads .env from cwd or parent dirs
print("ANTHROPIC_API_KEY loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="EV Explorer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Data paths ────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "outputs/processed_data"))
RAW_DIR       = Path(os.getenv("RAW_DIR",       "data/raw"))
# Resolve to absolute paths based on cwd at startup
if not PROCESSED_DIR.is_absolute():
    PROCESSED_DIR = (Path(__file__).parent / PROCESSED_DIR).resolve()
if not RAW_DIR.is_absolute():
    RAW_DIR = (Path(__file__).parent / RAW_DIR).resolve()


def _latest(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {directory}/{pattern}")
    return files[-1]


# ── Data loading (cached at startup) ─────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_vehicles() -> pd.DataFrame:
    try:
        path = _latest(PROCESSED_DIR, "epa_vehicles_*.csv")
    except FileNotFoundError:
        path = _latest(RAW_DIR, "epa_vehicles_*.csv")

    df = pd.read_csv(path, low_memory=False)

    # Keep only consumer BEVs (mirrors _bev_only in ev_analysis.py)
    fuel_col = "fuelType1" if "fuelType1" in df.columns else "fuel_type"
    if fuel_col in df.columns:
        df = df[df[fuel_col].str.strip() == "Electricity"]
    # Strict PHEV exclusion — must be False (not null, not True)
    if "is_phev" in df.columns:
        df = df[df["is_phev"].astype(str).str.strip().str.lower().isin(["false", "0", "no"])]
    if "VClass" in df.columns:
        commercial = ["Vans", "Vans, Cargo Type", "Vans, Passenger Type",
                      "Special Purpose Vehicles", "Special Purpose Vehicle 2WD",
                      "Special Purpose Vehicle 4WD"]
        df = df[~df["VClass"].str.contains("|".join(commercial), na=False, case=False)]
    # Strict MPGe floor — PHEVs and commercial vehicles score below 50
    if "combined_mpge" in df.columns:
        df = df[pd.to_numeric(df["combined_mpge"], errors="coerce") >= 50]

    # Convert cluster from float (2.0) to int (2), preserving NaN
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
        mask = df["cluster"].notna()
        df.loc[mask, "cluster"] = df.loc[mask, "cluster"].astype(int)

    # Friendly column aliases used by the frontend
    rename = {
        "fuelType1": "fuel_type", "fuelType2": "fuel_type2",
        "VClass": "vehicle_class", "trany": "transmission",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Synthetic vehicle ID for URL routing
    df = df.reset_index(drop=True)
    df["id"] = df.index

    return df


@lru_cache(maxsize=1)
def _load_stations() -> pd.DataFrame:
    try:
        path = _latest(PROCESSED_DIR, "charging_stations_*.csv")
    except FileNotFoundError:
        path = _latest(RAW_DIR, "charging_stations_*.csv")
    return pd.read_csv(path, low_memory=False)


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe records (handles NaN, numpy types)."""
    return json.loads(df.to_json(orient="records", default_handler=str))


# ── Haversine distance (km) ───────────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ── KPI summary ──────────────────────────────────────────────────────────────
def _build_summary(vehicles: pd.DataFrame, stations: pd.DataFrame) -> dict:
    return {
        "total_vehicles":      len(vehicles),
        "total_manufacturers": int(vehicles["make"].nunique()) if "make" in vehicles.columns else 0,
        "avg_range_miles":     round(float(vehicles["range_miles"].mean()), 1) if "range_miles" in vehicles.columns else None,
        "avg_mpge":            round(float(vehicles["combined_mpge"].mean()), 1) if "combined_mpge" in vehicles.columns else None,
        "total_stations":      len(stations),
        "dc_fast_stations":    int((stations["dc_fast_count"] > 0).sum()) if "dc_fast_count" in stations.columns else 0,
        "top_network":         stations["network"].value_counts().index[0] if "network" in stations.columns else None,
    }


# ── Segment summary ───────────────────────────────────────────────────────────
def _load_segment_stats() -> list[dict]:
    """Load pre-computed segment stats written by ev_analysis.py."""
    path = PROCESSED_DIR / "segment_stats.json"
    if path.exists():
        return json.loads(path.read_text())
    # Fallback: compute on the fly if file missing
    return _build_segments_fallback(_load_vehicles())


def _build_segments_fallback(vehicles: pd.DataFrame) -> list[dict]:
    """Fallback segment stats — less accurate, used only if segment_stats.json missing."""
    if "cluster" not in vehicles.columns:
        return []
    results = []
    for cid, grp in vehicles[vehicles["cluster"].notna()].groupby("cluster"):
        results.append({
            "cluster_id": int(cid),
            "count":      len(grp),
            "avg_mpge":   round(float(grp["combined_mpge"].mean()), 1) if "combined_mpge" in grp.columns else None,
            "avg_range":  round(float(grp["range_miles"].mean()), 1)   if "range_miles"   in grp.columns else None,
            "top_makes":  grp["make"].value_counts().head(3).index.tolist() if "make" in grp.columns else [],
        })
    return results


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    return _build_summary(_load_vehicles(), _load_stations())


@app.get("/api/vehicles")
def vehicles(
    make:       Optional[str]   = Query(None, description="Filter by manufacturer"),
    min_range:  Optional[float] = Query(None, description="Minimum range (miles)"),
    max_price:  Optional[float] = Query(None, description="Maximum base MSRP ($)"),
    min_mpge:   Optional[float] = Query(None, description="Minimum combined MPGe"),
    year:       Optional[int]   = Query(None, description="Model year"),
    min_year:   Optional[int]   = Query(None, description="Minimum model year"),
    cluster:    Optional[int]   = Query(None, description="GMM segment id"),
    has_battery: Optional[bool] = Query(None, description="Only vehicles with battery_capacity_kwh"),
    sort_by:    str             = Query("combined_mpge", description="Sort field"),
    sort_desc:  bool            = Query(True),
    limit:      int             = Query(200, le=2000),
    offset:     int             = Query(0),
):
    df = _load_vehicles().copy()

    if make:
        df = df[df["make"].str.lower() == make.lower()]
    if min_range is not None and "range_miles" in df.columns:
        df = df[df["range_miles"] >= min_range]
    if max_price is not None and "msrp_base" in df.columns:
        df = df[df["msrp_base"].notna() & (df["msrp_base"] <= max_price)]
    if min_mpge is not None and "combined_mpge" in df.columns:
        df = df[df["combined_mpge"] >= min_mpge]
    if year is not None and "year" in df.columns:
        df = df[df["year"] == year]
    if min_year is not None and "year" in df.columns:
        df = df[df["year"] >= min_year]
    if cluster is not None and "cluster" in df.columns:
        df = df[df["cluster"] == cluster]
    if has_battery and "battery_capacity_kwh" in df.columns:
        df = df[df["battery_capacity_kwh"].notna()]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")

    total = len(df)
    page  = df.iloc[offset : offset + limit]

    # Only send columns the frontend needs
    cols = [c for c in [
        "id", "year", "make", "model", "vehicle_class", "transmission", "drive",
        "combined_mpge", "city_mpge", "highway_mpge", "range_miles",
        "battery_capacity_kwh", "kwh_per_100mi", "msrp_base", "annual_fuel_cost_usd",
        "connector_type", "fast_charge_minutes", "charge_240v_hrs",
        "max_dc_kw", "acceleration_0_60", "cluster",
        *[c for c in df.columns if c.startswith("cluster_proba_")],
    ] if c in page.columns]

    return {
        "total":   total,
        "offset":  offset,
        "limit":   limit,
        "results": _df_to_records(page[cols]),
    }


@app.get("/api/vehicles/makes")
def makes():
    df = _load_vehicles()
    counts = df["make"].value_counts().reset_index()
    counts.columns = ["make", "count"]
    return _df_to_records(counts)


@app.get("/api/vehicles/{vehicle_id}")
def vehicle_detail(vehicle_id: int):
    df = _load_vehicles()
    row = df[df["id"] == vehicle_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return _df_to_records(row)[0]


@app.get("/api/stations/nearby")
def stations_nearby(
    lat:          float = Query(..., description="Latitude"),
    lon:          float = Query(..., description="Longitude"),
    radius_km:    float = Query(25, description="Search radius in km"),
    dc_fast_only: bool  = Query(False),
    limit: Optional[int] = Query(None, le=50000),
):
    df = _load_stations().copy()
    df = df.dropna(subset=["latitude", "longitude"])

    if dc_fast_only and "dc_fast_count" in df.columns:
        df = df[df["dc_fast_count"] > 0]

    # Vectorised haversine — ~100x faster than row-by-row .apply()
    R = 6371.0
    lat1, lon1 = radians(lat), radians(lon)
    phi2 = np.radians(df["latitude"].to_numpy())
    lam2 = np.radians(df["longitude"].to_numpy())
    dphi = phi2 - lat1
    dlam = lam2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    df["distance_km"] = 2 * R * np.arcsin(np.sqrt(a))

    nearby = df[df["distance_km"] <= radius_km].sort_values("distance_km")
    if limit:
        nearby = nearby.head(limit)

    cols = [c for c in [
        "station_id", "station_name", "city", "state", "network",
        "access_code", "latitude", "longitude", "distance_km",
        "level1_count", "level2_count", "dc_fast_count", "access_hours",
    ] if c in nearby.columns]

    return _df_to_records(nearby[cols])


@app.get("/api/stations/networks")
def networks():
    df = _load_stations()
    counts = df["network"].value_counts().reset_index()
    counts.columns = ["network", "count"]
    total = counts["count"].sum()
    counts["share_pct"] = (counts["count"] / total * 100).round(1)
    return _df_to_records(counts.head(15))


@app.get("/api/stations")
def stations(
    city:         Optional[str]   = Query(None),
    state:        Optional[str]   = Query(None),
    network:      Optional[str]   = Query(None),
    dc_fast_only: bool            = Query(False),
    lat_min:      Optional[float] = Query(None, description="Bounding box south"),
    lat_max:      Optional[float] = Query(None, description="Bounding box north"),
    lon_min:      Optional[float] = Query(None, description="Bounding box west"),
    lon_max:      Optional[float] = Query(None, description="Bounding box east"),
    limit:        int             = Query(500, le=5000),
    offset:       int             = Query(0),
):
    df = _load_stations().copy()
    df = df.dropna(subset=["latitude", "longitude"])

    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if state:
        df = df[df["state"].str.upper() == state.upper()]
    if network:
        df = df[df["network"].str.lower().str.contains(network.lower(), na=False)]
    if dc_fast_only and "dc_fast_count" in df.columns:
        df = df[df["dc_fast_count"] > 0]
    if lat_min is not None:
        df = df[df["latitude"] >= lat_min]
    if lat_max is not None:
        df = df[df["latitude"] <= lat_max]
    if lon_min is not None:
        df = df[df["longitude"] >= lon_min]
    if lon_max is not None:
        df = df[df["longitude"] <= lon_max]

    total = len(df)
    page  = df.iloc[offset : offset + limit]

    cols = [c for c in [
        "station_id", "station_name", "city", "state", "network",
        "access_code", "latitude", "longitude",
        "level1_count", "level2_count", "dc_fast_count",
        "connector_types", "access_hours",
    ] if c in page.columns]

    return {
        "total":   total,
        "offset":  offset,
        "limit":   limit,
        "results": _df_to_records(page[cols]),
    }


@app.get("/api/segments")
def segments():
    return _load_segment_stats()


@app.get("/api/segments/vehicles")
def segments_vehicles():
    """Deduplicated vehicles for segment scatter — one dot per base model."""
    import re as _re
    _TRIM = _re.compile(
        r"\b(long range|standard range|standard|performance|extended range|extended|"
        r"plus|pro|max|ultra|premium|limited|elite|gt|sport|turbo s|turbo|plaid|"
        r"awd|rwd|fwd|4wd|4x4|dual motor|single motor|tri motor|"
        r"cross turismo|sportback|avant|allroad|coupe|cabriolet|convertible|"
        r"\d+in|\d+\s*inch|\(.*\)|\d+[dwh]|kwh|kw)\b.*",
        _re.IGNORECASE
    )
    def base_name(m: str) -> str:
        m = _TRIM.sub("", str(m)).strip()
        return _re.sub(r"  +", " ", m).strip().lower()

    df = _load_vehicles().copy()
    df = df[df["cluster"].notna()]
    # Only current-market vehicles (2022+) — older rows have stale segment assignments
    if "year" in df.columns:
        df = df[df["year"] >= 2022]
    df["_base"] = df["model"].apply(base_name)
    df["_mk"]   = df["make"].str.strip().str.lower()
    if "year" in df.columns:
        df = df.sort_values("year", ascending=False).drop_duplicates(subset=["_mk", "_base"])
    df = df.drop(columns=["_base", "_mk"], errors="ignore")
    cols = [c for c in [
        "id", "year", "make", "model", "combined_mpge", "range_miles", "cluster",
    ] if c in df.columns]
    return _df_to_records(df[cols])


@app.get("/api/trends")
def trends():
    """Manufacturer efficiency trend data for the chart."""
    df = _load_vehicles()
    if "make" not in df.columns or "year" not in df.columns:
        return []

    yearly = (
        df.groupby(["make", "year"])["combined_mpge"]
        .mean()
        .reset_index()
        .rename(columns={"combined_mpge": "avg_mpge"})
    )
    yearly["avg_mpge"] = yearly["avg_mpge"].round(1)
    return _df_to_records(yearly)

# ── Claude batch normalizer ───────────────────────────────────────────────────
from anthropic import AsyncAnthropic
from pydantic import BaseModel as _BaseModel

_anthropic = AsyncAnthropic()

BASE_MODEL_MAP_PATH = PROCESSED_DIR / "base_model_map.json"
_NORMALIZE_BATCH_SIZE = 50

_NORMALIZE_PROMPT = """\
You are normalizing EV model names for a car comparison app.
Given the list of raw model name strings below, return a single JSON object where:
- Each KEY is the exact raw model name string (copied verbatim)
- Each VALUE is the clean base model name with ALL of the following removed:
  trim levels (Long Range, Performance, Plus, Pro, Max, Limited, GT, Sport, Plaid, Elite, Ultra, Premium…)
  drivetrain codes (AWD, RWD, FWD, 4WD, Dual Motor, Single Motor, E-4orce, Quattro, xDrive, 4Motion…)
  battery/range descriptors (63kWh, 87kWh, Standard Range, Extended Range, Long Range…)
  variant suffixes (Engage+, Evolve+, Connect+, Sportback, Avant, Allroad, Cross Turismo…)
  anything in parentheses, trailing punctuation, or stray characters

Keep only the core model identity — what a customer would call the car at a dealership.
Preserve original capitalisation of the base name.
Return ONLY valid JSON, no explanation, no markdown fences.

Raw model names:
"""


async def _normalize_batch(names: list[str]) -> dict[str, str]:
    """Call Claude for one batch; returns {rawName: baseName}."""
    prompt = _NORMALIZE_PROMPT + "\n".join(f'"{n}"' for n in names)
    message = await _anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=8096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


async def build_base_model_map() -> dict[str, str]:
    """
    Chunk all unique model names into batches of _NORMALIZE_BATCH_SIZE,
    call Claude for each batch, and merge into one map.
    Skips entirely if base_model_map.json already exists.
    """
    if BASE_MODEL_MAP_PATH.exists():
        print(f"[startup] base_model_map.json already exists — skipping normalization.")
        return json.loads(BASE_MODEL_MAP_PATH.read_text())

    df = _load_vehicles()
    all_names = sorted(df["model"].dropna().unique().tolist())
    print(f"[startup] Normalizing {len(all_names)} unique model names "
          f"in batches of {_NORMALIZE_BATCH_SIZE}…")

    merged: dict[str, str] = {}
    batches = [all_names[i:i + _NORMALIZE_BATCH_SIZE]
               for i in range(0, len(all_names), _NORMALIZE_BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        print(f"[startup]   batch {idx}/{len(batches)} ({len(batch)} names)…")
        try:
            result = await _normalize_batch(batch)
            merged.update(result)
        except Exception as exc:
            # On partial failure keep going; missing keys fall back to regex in the UI
            print(f"[startup]   WARNING: batch {idx} failed — {exc}")

    BASE_MODEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASE_MODEL_MAP_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"[startup] Saved {len(merged)} entries → {BASE_MODEL_MAP_PATH}")
    return merged


@app.on_event("startup")
async def startup_event():
    try:
        await build_base_model_map()
    except Exception as exc:
        # Don't crash the server if normalization fails; UI has a regex fallback
        print(f"[startup] base_model_map build failed: {exc}")


@app.get("/api/claude/base-model-map")
async def base_model_map():
    """Return the pre-built {rawModelName: baseName} map built at startup."""
    if not BASE_MODEL_MAP_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="base_model_map.json not ready yet — server may still be normalizing.",
        )
    return json.loads(BASE_MODEL_MAP_PATH.read_text())


# Kept for manual re-generation (e.g. after dataset updates). Not called by the frontend.
class _ClaudeRequest(_BaseModel):
    models: list[str]

@app.post("/api/claude/normalize-models")
async def normalize_models(body: _ClaudeRequest):
    """Re-run normalization for an explicit list of names and overwrite the cache."""
    result = await _normalize_batch(body.models)
    # Merge into existing map (don't throw away cached entries)
    existing = json.loads(BASE_MODEL_MAP_PATH.read_text()) if BASE_MODEL_MAP_PATH.exists() else {}
    existing.update(result)
    BASE_MODEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASE_MODEL_MAP_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return result

# ── Serve React build (must be last) ─────────────────────────────────────────
FRONTEND_BUILD = Path("frontend/dist")
if FRONTEND_BUILD.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="static")