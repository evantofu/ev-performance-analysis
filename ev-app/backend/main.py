"""Read-only API for the bundled EV and charging-station snapshot."""

from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.vehicle_rules import base_model_name, consumer_bevs, manufacturer_trends, segment_vehicles  # noqa: E402

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT / "outputs/processed_data")))
FRONTEND_BUILD = (
    HERE / "frontend/dist" if (HERE / "frontend/dist").exists() else HERE.parent / "frontend/dist"
)
app = FastAPI(title="EV Explorer API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _latest(pattern: str) -> Path:
    files = sorted(PROCESSED_DIR.glob(pattern))
    if not files:
        raise HTTPException(503, detail="Dataset unavailable; check PROCESSED_DIR or rebuild the snapshot")
    return files[-1]


@lru_cache(maxsize=1)
def _load_vehicles() -> pd.DataFrame:
    df = consumer_bevs(pd.read_csv(_latest("epa_vehicles_*.csv"), low_memory=False))
    df, _, _ = segment_vehicles(df)
    df = df.rename(columns={"VClass": "vehicle_class", "trany": "transmission"})
    df["base_model"] = df["model"].map(base_model_name)
    # Preserve EPA identifiers across filtering, sorting, and snapshot rebuilds.
    df["id"] = pd.to_numeric(df["fueleconomy_id"], errors="raise").astype(int)
    return df


@lru_cache(maxsize=1)
def _load_stations() -> pd.DataFrame:
    df = pd.read_csv(_latest("charging_stations_*.csv"), low_memory=False)
    for field in ["latitude", "longitude", "dc_fast_count"]:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    return df


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", default_handler=str))


def _page(df: pd.DataFrame, limit: int, offset: int, columns: list[str]) -> dict:
    return {
        "total": len(df),
        "offset": offset,
        "limit": limit,
        "results": _df_to_records(df.iloc[offset : offset + limit][[c for c in columns if c in df]]),
    }


VEHICLE_FIELDS = [
    "id",
    "base_model",
    "year",
    "make",
    "model",
    "vehicle_class",
    "transmission",
    "drive",
    "combined_mpge",
    "city_mpge",
    "highway_mpge",
    "range_miles",
    "battery_capacity_kwh",
    "kwh_per_100mi",
    "msrp_base",
    "annual_fuel_cost_usd",
    "connector_type",
    "fast_charge_minutes",
    "charge_240v_hrs",
    "max_ac_kw",
    "max_dc_kw",
    "acceleration_0_60",
    "cluster",
]
STATION_FIELDS = [
    "station_id",
    "station_name",
    "city",
    "state",
    "network",
    "access_code",
    "latitude",
    "longitude",
    "level1_count",
    "level2_count",
    "dc_fast_count",
    "connector_types",
    "access_hours",
    "distance_km",
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ok", "vehicles": len(_load_vehicles()), "stations": len(_load_stations())}


@app.get("/api/summary")
def summary():
    vehicles, stations = _load_vehicles(), _load_stations()
    meta_files = sorted(PROCESSED_DIR.glob("metadata_*.json"))
    meta = json.loads(meta_files[-1].read_text()) if meta_files else {}
    return {
        "total_vehicles": len(vehicles),
        "total_manufacturers": vehicles["make"].nunique(),
        "avg_range_miles": round(float(vehicles["range_miles"].mean()), 1),
        "avg_mpge": round(float(vehicles["combined_mpge"].mean()), 1),
        "total_stations": len(stations),
        "dc_fast_stations": int(stations["dc_fast_count"].gt(0).sum()),
        "snapshot_date": meta.get("export_date"),
        "latest_model_year": int(vehicles["year"].max()),
    }


@app.get("/api/vehicles")
def vehicles(
    make: str | None = None,
    min_range: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    min_mpge: float | None = Query(None, ge=0),
    year: int | None = None,
    min_year: int | None = None,
    cluster: int | None = Query(None, ge=0, le=3),
    has_battery: bool = False,
    dedup_models: bool = False,
    sort_by: str = Query("combined_mpge", pattern="^(combined_mpge|range_miles|year|msrp_base)$"),
    sort_desc: bool = True,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    df = _load_vehicles()
    if make:
        df = df.loc[df["make"].str.casefold().eq(make.casefold())]
    for field, value in [("range_miles", min_range), ("combined_mpge", min_mpge), ("year", min_year)]:
        if value is not None:
            df = df.loc[df[field].ge(value)]
    if year is not None:
        df = df.loc[df["year"].eq(year)]
    if max_price is not None:
        df = df.loc[df["msrp_base"].le(max_price)]
    if cluster is not None:
        df = df.loc[df["cluster"].eq(cluster).fillna(False)]
    if has_battery:
        df = df.loc[df["battery_capacity_kwh"].notna()]
    if dedup_models:
        df = df.sort_values(["year", "id"], ascending=[False, True]).drop_duplicates(["make", "base_model"])
    df = df.sort_values([sort_by, "id"], ascending=[not sort_desc, True], na_position="last")
    return _page(df, limit, offset, VEHICLE_FIELDS)


@app.get("/api/vehicles/makes")
def makes():
    return _df_to_records(
        _load_vehicles()["make"].value_counts().rename_axis("make").reset_index(name="count")
    )


@app.get("/api/vehicles/{vehicle_id}")
def vehicle_detail(vehicle_id: int):
    row = _load_vehicles().loc[lambda d: d["id"].eq(vehicle_id)]
    if row.empty:
        raise HTTPException(404, detail="Vehicle not found")
    return _df_to_records(row[[c for c in VEHICLE_FIELDS if c in row]])[0]


@app.get("/api/stations/networks")
def networks():
    counts = _load_stations()["network"].value_counts().rename_axis("network").reset_index(name="count")
    counts["share_pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    return _df_to_records(counts.head(15))


@app.get("/api/stations/nearby")
def stations_nearby(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(25, gt=0, le=500),
    dc_fast_only: bool = False,
    limit: int = Query(200, ge=1, le=1000),
):
    df = _load_stations().dropna(subset=["latitude", "longitude"]).copy()
    if dc_fast_only:
        df = df.loc[df["dc_fast_count"].gt(0)].copy()
    phi = np.radians(df["latitude"].to_numpy())
    dphi = phi - np.radians(lat)
    dlon = np.radians(df["longitude"].to_numpy() - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(phi) * np.sin(dlon / 2) ** 2
    df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    df = df.loc[df["distance_km"].le(radius_km)].sort_values("distance_km").head(limit)
    return _df_to_records(df[[c for c in STATION_FIELDS if c in df]])


@app.get("/api/stations")
def stations(
    city: str | None = None,
    state: str | None = None,
    network: str | None = None,
    dc_fast_only: bool = False,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    lat_min: float | None = Query(None, ge=-90, le=90),
    lat_max: float | None = Query(None, ge=-90, le=90),
    lon_min: float | None = Query(None, ge=-180, le=180),
    lon_max: float | None = Query(None, ge=-180, le=180),
):
    bounds = [lat_min, lat_max, lon_min, lon_max]
    if any(b is not None for b in bounds) and not all(b is not None for b in bounds):
        raise HTTPException(422, detail="Supply all four viewport bounds")
    df = _load_stations()
    if lat_min is not None:
        if lat_min > lat_max:
            raise HTTPException(422, detail="lat_min must be <= lat_max")
        longitude = (
            df["longitude"].between(lon_min, lon_max)
            if lon_min <= lon_max
            else (df["longitude"].ge(lon_min) | df["longitude"].le(lon_max))
        )
        df = df.loc[df["latitude"].between(lat_min, lat_max) & longitude]
    for field, value in [("city", city), ("state", state)]:
        if value:
            df = df.loc[df[field].str.casefold().eq(value.casefold())]
    if network:
        df = df.loc[df["network"].str.contains(network, case=False, na=False, regex=False)]
    if dc_fast_only:
        df = df.loc[df["dc_fast_count"].gt(0)]
    return _page(df, limit, offset, STATION_FIELDS)


@app.get("/api/segments")
def segments():
    return segment_vehicles(_load_vehicles())[2]


@app.get("/api/segments/vehicles")
def segments_vehicles():
    df = _load_vehicles().dropna(subset=["cluster"])
    df = df.sort_values(["year", "id"], ascending=[False, True]).drop_duplicates(["make", "base_model"])
    return _df_to_records(df[[c for c in VEHICLE_FIELDS if c in df]])


@app.get("/api/trends")
def trends():
    yearly, fits = manufacturer_trends(_load_vehicles())
    result = yearly.rename(columns={"combined_mpge": "avg_mpge"}).merge(fits, on="make")
    result["avg_mpge"] = result["avg_mpge"].round(1)
    return _df_to_records(result)


# SPA fallback supports directly opening /compare, while unknown API paths stay 404.
if FRONTEND_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_BUILD / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, detail="API route not found")
        candidate = (FRONTEND_BUILD / path).resolve()
        if candidate.is_relative_to(FRONTEND_BUILD.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_BUILD / "index.html")
