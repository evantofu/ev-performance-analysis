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
# Path resolution works for both local dev and Docker:
# - Docker: /app/outputs/processed_data (copied by Dockerfile)
# - Local:  repo_root/outputs/processed_data (3 levels up from ev-app/backend/)
_HERE      = Path(__file__).parent.resolve()
_REPO_ROOT = _HERE.parent.parent  # ev-app/backend → ev-app → repo root
_DOCKER_DATA = _HERE / "outputs" / "processed_data"  # Docker path

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR",
    str(_DOCKER_DATA if _DOCKER_DATA.exists()
        else _REPO_ROOT / "outputs" / "processed_data")))
RAW_DIR = Path(os.getenv("RAW_DIR",
    str(_HERE / "data" / "raw" if (_HERE / "data" / "raw").exists()
        else _REPO_ROOT / "data" / "raw")))


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
    has_battery:  Optional[bool] = Query(None, description="Only vehicles with battery_capacity_kwh"),
    dedup_models: bool           = Query(False, description="One row per base model (latest year)"),
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

    # dedup_models: collapse to one row per make+base_model (latest year)
    # This is what consumers actually want — one card per car, not per trim
    if dedup_models and "model" in df.columns:
        import re as _re
        _TRIM = _re.compile(
            r"\b(long range|standard range|standard|performance|extended range|extended|"
            r"plus|pro|max|ultra|premium|limited|elite|gt|sport|turbo s|turbo|plaid|"
            r"awd|rwd|fwd|4wd|dual motor|single motor|tri motor|"
            r"cross turismo|sportback|avant|allroad|coupe|cabriolet|"
            r"\d+in|\d+\s*inch|\(.*?\)|\d+[dwh]|kwh|kw)\b.*",
            _re.IGNORECASE
        )
        def _base(m):
            m = _TRIM.sub("", str(m)).strip()
            return _re.sub(r"  +", " ", m).strip().lower()
        df["_base"] = df["model"].apply(_base)
        df["_mk"]   = df["make"].str.strip().str.lower()
        df = df.sort_values("year", ascending=False).drop_duplicates(subset=["_mk", "_base"])
        df = df.drop(columns=["_base", "_mk"], errors="ignore")
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
    limit:        int   = Query(200, le=1000),
):
    df = _load_stations().copy()
    df = df.dropna(subset=["latitude", "longitude"])

    if dc_fast_only and "dc_fast_count" in df.columns:
        df = df[df["dc_fast_count"] > 0]

    df["distance_km"] = df.apply(
        lambda r: _haversine(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    nearby = df[df["distance_km"] <= radius_km].sort_values("distance_km").head(limit)

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
    city:        Optional[str]  = Query(None),
    state:       Optional[str]  = Query(None),
    network:     Optional[str]  = Query(None),
    dc_fast_only: bool          = Query(False),
    limit:       int            = Query(500, le=5000),
    offset:      int            = Query(0),
):
    df = _load_stations().copy()

    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if state:
        df = df[df["state"].str.upper() == state.upper()]
    if network:
        df = df[df["network"].str.lower().str.contains(network.lower(), na=False)]
    if dc_fast_only and "dc_fast_count" in df.columns:
        df = df[df["dc_fast_count"] > 0]

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


# ── Serve React build (must be last) ─────────────────────────────────────────
FRONTEND_BUILD = Path("frontend/dist")
if FRONTEND_BUILD.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="static")