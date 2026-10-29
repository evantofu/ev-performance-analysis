"""Shared, deterministic vehicle identity, filtering, and segment definitions."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAMES = json.loads(Path(__file__).with_name("model_names.json").read_text())
SEGMENT_NAMES = {0: "High Efficiency", 1: "Mainstream", 2: "Longer Range", 3: "Lower Efficiency"}
SEGMENT_MIN_YEAR = 2022
_TRIM = re.compile(
    r"\b(?:long range|standard range|performance|extended range|premium|limited|"
    r"dual motor|single motor|tri motor|awd|rwd|fwd|4wd|4x4|quattro|xdrive|"
    r"4motion|plaid|\d+\s*(?:inch|in)\b|\d+\s*kwh\b)\b.*",
    re.I,
)


def base_model_name(model: str) -> str:
    """Use the bundled lookup; preserve unknown names unless a known trim matches."""
    raw = str(model).strip()
    clean = MODEL_NAMES.get(raw) or _TRIM.sub("", raw).strip()
    return re.sub(r"\s+", " ", clean or raw).lower()


def consumer_bevs(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Filter the consumer cohort while retaining the original index for joins.

    The 50 MPGe floor is a project cohort rule, not a definition of a BEV.
    Missing fuel identity is rejected rather than treated as electric.
    """
    fuel = next((c for c in ("fuelType1", "fuel_type") if c in vehicles), None)
    if fuel is None:
        return vehicles.iloc[:0].copy()
    bev = vehicles.loc[vehicles[fuel].str.strip().eq("Electricity")].copy()
    if "atvType" in bev:
        bev = bev.loc[~bev["atvType"].str.contains("PHEV|Plug-in", case=False, na=False)]
    if "is_phev" in bev:
        bev = bev.loc[bev["is_phev"].astype(str).str.lower().str.strip().isin(["false", "0", "no"])]
    if "model" in bev:
        bev = bev.loc[~bev["model"].str.contains(r"plug.in|phev|e-tron\s+\d+e\b", case=False, na=False)]
    vclass = next((c for c in ("VClass", "vehicle_class") if c in bev), None)
    if vclass:
        bev = bev.loc[~bev[vclass].str.contains("Vans|Special Purpose Vehicle", case=False, na=False)]
    if "combined_mpge" in bev:
        bev = bev.loc[pd.to_numeric(bev["combined_mpge"], errors="coerce").ge(50)]
    return bev


def segment_vehicles(vehicles: pd.DataFrame) -> tuple[pd.DataFrame, int, list[dict]]:
    """Label the 2022+ cohort using latest-model-year base-model averages.

    Thresholds are the 33rd/67th efficiency and 67th range percentiles.
    Efficiency takes precedence; the remaining upper-range models get label 2.
    These are descriptive rules, not learned clusters or purchase recommendations.
    """
    out = vehicles.copy()
    out["cluster"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    required = {"make", "model", "year", "combined_mpge", "range_miles"}
    if not required.issubset(out.columns):
        return out, 0, []
    bev = consumer_bevs(out)
    for col in ("year", "combined_mpge", "range_miles"):
        bev[col] = pd.to_numeric(bev[col], errors="coerce")
    numeric = ["year", "combined_mpge", "range_miles"]
    bev[numeric] = bev[numeric].replace([np.inf, -np.inf], np.nan)
    bev = bev.dropna(subset=list(required))
    bev = bev.loc[(bev["year"] >= SEGMENT_MIN_YEAR) & (bev["range_miles"] > 0)].copy()
    bev["_make"] = bev["make"].str.strip().str.lower()
    bev["_base"] = bev["model"].map(base_model_name)
    latest_year = bev.groupby(["_make", "_base"])["year"].transform("max")
    latest = bev.loc[bev["year"].eq(latest_year)]
    agg = latest.groupby(["_make", "_base"]).agg(mpge=("combined_mpge", "mean"), rng=("range_miles", "mean"))
    if len(agg) < 20:
        return out, 0, []
    low, high = agg["mpge"].quantile([0.33, 0.67])
    range_high = agg["rng"].quantile(0.67)
    agg["cluster"] = np.select(
        [agg["mpge"] >= high, agg["mpge"] < low, agg["rng"] >= range_high],
        [0, 3, 2],
        default=1,
    )
    keys = pd.MultiIndex.from_frame(bev[["_make", "_base"]])
    out.loc[bev.index, "cluster"] = agg["cluster"].reindex(keys).to_numpy()
    stats = []
    for cid, grp in agg.groupby("cluster"):
        members = out.loc[out["cluster"].eq(cid).fillna(False)]
        stats.append(
            {
                "cluster_id": int(cid),
                "count": len(members),
                "base_model_count": len(grp),
                "avg_mpge": round(float(grp["mpge"].mean()), 1),
                "avg_range": round(float(grp["rng"].mean()), 1),
                "top_makes": members["make"].value_counts().head(3).index.tolist(),
            }
        )
    return out, len(stats), stats


def manufacturer_trends(vehicles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return yearly means and OLS diagnostics for makes with >=4 years and R²>=0.3.

    R² describes in-sample linear fit; it does not measure forecast accuracy.
    """
    yearly = consumer_bevs(vehicles).groupby(["make", "year"])["combined_mpge"].mean().reset_index()
    yearly = yearly.replace([np.inf, -np.inf], np.nan).dropna()
    diagnostics = []
    for make, grp in yearly.groupby("make"):
        if len(grp) < 4:
            continue
        x = grp["year"].to_numpy(dtype=float)
        y = grp["combined_mpge"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(((y - (slope * x + intercept)) ** 2).sum()) / total if total > 0 else 0
        if r2 >= 0.3:
            diagnostics.append({"make": make, "slope": slope, "r2": r2, "years": len(grp)})
    fits = pd.DataFrame(diagnostics, columns=["make", "slope", "r2", "years"])
    return yearly.loc[yearly["make"].isin(fits["make"])], fits
