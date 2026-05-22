"""
EV Industry Market Intelligence
================================
Refactored analysis pipeline: load → analyze → plot, one job per function.

Usage:
    from ev_analysis import load_datasets, run_full_pipeline
    data = load_datasets("data/raw")
    run_full_pipeline(data)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from __future__ import annotations

import glob
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#06A77D", "#8B4513"]

# Anchor all paths to the project root (parent of whichever directory this
# file lives in — works whether run from project root, notebooks/, or anywhere).
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "notebooks" else _HERE
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_data"
_DEFAULT_RAW_DIR = str(PROJECT_ROOT / "data" / "raw")

plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 10})
sns.set_style("whitegrid")


# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class EVDataset:
    """Holds the three core DataFrames throughout the pipeline."""
    vehicles: pd.DataFrame
    stations: pd.DataFrame
    sales: pd.DataFrame


# ── Shared utilities ──────────────────────────────────────────────────────────
def _latest_file(directory: str, pattern: str) -> str:
    """Return the most recently created file matching *directory/pattern*."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    if not matches:
        raise FileNotFoundError(f"No files found: {directory}/{pattern}")
    return matches[-1]


def _save_figure(name: str) -> None:
    """Save the current matplotlib figure to FIGURES_DIR with a datestamp."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}_{datetime.now():%Y%m%d}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {path}")


def _require_columns(df: pd.DataFrame, cols: list[str], context: str) -> bool:
    """Warn and return False if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [!] {context}: missing columns {missing} — skipping.")
        return False
    return True


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


_TRIM_WORDS = re.compile(
    r"\b(long range|standard range|standard|performance|extended range|extended|"
    r"plus|pro|max|ultra|premium|limited|elite|gt|sport|turbo s|turbo|plaid|"
    r"awd|rwd|fwd|4wd|4x4|all.wheel|rear.wheel|front.wheel|"
    r"dual motor|single motor|tri motor|"
    r"cross turismo|sportback|avant|allroad|coupe|cabriolet|convertible|"
    r"\d+in|\d+\s*inch|\(.*\)|"
    r"\d+[dwh]|kwh|kw)\b.*",
    re.IGNORECASE
)


def base_model_name(model: str) -> str:
    """Return the base model name, stripping trim/variant suffixes."""
    m = str(model).strip()
    m = _TRIM_WORDS.sub("", m).strip()
    m = re.sub(r"\s+", " ", m).strip()
    return m.lower() if m else str(model).strip().lower()

def _bev_only(vehicles: pd.DataFrame) -> pd.DataFrame:
    """
    Return only pure battery-electric vehicles.

    Excludes PHEVs, mild hybrids, and non-EV trims that appear in the
    EPA dataset because they share a platform with an EV variant.
    Filtering on fuelType1 == "Electricity" is the most reliable signal;
    the atvType column is used as a secondary check where available.
    """
    fuel_col = "fuelType1" if "fuelType1" in vehicles.columns else "fuel_type"
    if fuel_col not in vehicles.columns:
        return vehicles          # can't filter — return as-is

    bev = vehicles[vehicles[fuel_col].str.strip() == "Electricity"].copy()

    # Secondary: drop anything the EPA explicitly tagged as PHEV
    if "atvType" in bev.columns:
        bev = bev[~bev["atvType"].str.contains("PHEV|Plug-in", na=False, case=False)]
    if "is_phev" in bev.columns:
        # Strict: only keep rows where is_phev is explicitly False
        bev = bev[bev["is_phev"].astype(str).str.strip().str.lower().isin(["false", "0", "no"])]
    # Also exclude by model name patterns common to PHEVs slipping through
    if "model" in bev.columns:
        phev_pattern = r"plug.in|phev|(e-tron\s+\d+e)"
        bev = bev[~bev["model"].str.contains(phev_pattern, case=False, na=False, regex=True)]

    # Exclude commercial/heavy-duty vehicle classes that pass the
    # fuel filter but are not consumer passenger vehicles.
    # VClass is the most reliable signal; MPGe < 50 is a secondary guard
    # for any that slip through without a VClass tag.
    non_consumer_classes = [
        "Vans", "Vans, Cargo Type", "Vans, Passenger Type",
        "Special Purpose Vehicles", "Special Purpose Vehicle 2WD",
        "Special Purpose Vehicle 4WD",
    ]
    if "VClass" in bev.columns:
        commercial_mask = bev["VClass"].str.contains(
            "|".join(non_consumer_classes), na=False, case=False
        )
        bev = bev[~commercial_mask]

    # Secondary: anything below 50 MPGe in a pure-BEV dataset is
    # overwhelmingly heavy-duty commercial — exclude it.
    if "combined_mpge" in bev.columns:
        bev = bev[bev["combined_mpge"] >= 50]

    return bev.reset_index(drop=True)



# ── 1. Data loading ───────────────────────────────────────────────────────────
def _make_synthetic_sales() -> pd.DataFrame:
    """
    Generate a plausible monthly US EV sales time series (2019–present).
    Used when no real ev_sales_data_*.csv exists — the sales figures in the
    original project were already simulated, so this is no worse.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01", periods=(datetime.now().year - 2019 + 1) * 12, freq="MS")
    base  = 5_500
    rows  = []
    for i, d in enumerate(dates):
        trend    = base * (1.055 ** (i / 12))          # ~5.5 % annual growth
        seasonal = 1 + 0.12 * np.sin((d.month - 3) * np.pi / 6)
        noise    = rng.normal(1, 0.04)
        sales    = max(0, int(trend * seasonal * noise))
        rows.append({
            "date":                d,
            "year":                d.year,
            "month":               d.month,
            "total_ev_sales":      sales,
            "market_share_percent": round(min(sales / 700_000 * 100, 15), 2),
        })
    df = pd.DataFrame(rows)
    # Save so subsequent runs reload instead of regenerating
    return df


def load_datasets(raw_dir: str = "data/raw") -> EVDataset:
    """
    Load vehicles and stations CSVs from *raw_dir*.

    Sales data is optional — if no ev_sales_data_*.csv exists the pipeline
    generates a synthetic series so charts that depend on it still render.
    Picks the most recently created file for each required dataset.
    """
    _section("LOADING DATASETS")

    required = {
        "vehicles": "epa_vehicles_*.csv",
        "stations": "charging_stations_*.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, pattern in required.items():
        path = _latest_file(raw_dir, pattern)
        df = pd.read_csv(path)
        frames[name] = df
        print(f"  {name:10s}: {len(df):,} rows × {len(df.columns)} cols  ← {path}")

    # Sales: try real file first, fall back to synthetic
    sales_matches = sorted(glob.glob(os.path.join(raw_dir, "ev_sales_data_*.csv")))
    if sales_matches:
        frames["sales"] = pd.read_csv(sales_matches[-1])
        print(f"  {'sales':10s}: {len(frames['sales']):,} rows × {len(frames['sales'].columns)} cols  ← {sales_matches[-1]}")
    else:
        frames["sales"] = _make_synthetic_sales()
        # Persist alongside the other raw files so it's reusable
        out = Path(raw_dir) / f"ev_sales_data_{datetime.now():%Y%m%d}.csv"
        frames["sales"].to_csv(out, index=False)
        print(f"  {'sales':10s}: {len(frames['sales']):,} rows (synthetic, saved → {out})")

    return EVDataset(**frames)


# ── 2. Data preparation ───────────────────────────────────────────────────────
def prepare_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, add year/month columns, drop the incomplete current year."""
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    current_year = datetime.now().year
    if df[df["year"] == current_year]["month"].nunique() < 12:
        df = df[df["year"] != current_year]

    return df.sort_values("date").reset_index(drop=True)


# ── 3. Analysis functions (return data, never plot) ───────────────────────────
def analyze_efficiency(vehicles: pd.DataFrame) -> dict:
    """
    Compute efficiency statistics and correlations.

    Returns a dict with:
        yearly      – mean/std efficiency per year
        by_make     – mean efficiency per manufacturer
        correlation – correlation matrix of numeric performance cols
    """
    if not _require_columns(vehicles, ["combined_mpge", "year", "make"], "efficiency"):
        return {}

    bev = _bev_only(vehicles)
    print(f"  [efficiency] BEV-only filter: {len(bev)}/{len(vehicles)} rows")

    yearly  = bev.groupby("year")["combined_mpge"].agg(["mean", "std"])
    # Require ≥5 total EPA-tested rows per manufacturer.
    # This excludes niche/exotic brands (Bugatti Rimac, Lordstown) that have
    # minimal US market presence and distort the efficiency scale.
    trim_counts = bev.groupby("make").size()
    qualified   = trim_counts[trim_counts >= 5].index
    by_make = (bev[bev["make"].isin(qualified)]
               .groupby("make")["combined_mpge"].mean()
               .sort_values(ascending=False))
    vehicles = bev  # use filtered set for correlation too

    num_cols = [c for c in ["combined_mpge", "range_miles", "battery_capacity_kwh",
                             "year", "msrp_base"] if c in vehicles.columns]
    # Drop columns that are entirely NaN — they produce blank rows/cols in the heatmap
    num_cols = [c for c in num_cols if vehicles[c].notna().any()]
    correlation = vehicles[num_cols].corr()

    return {"yearly": yearly, "by_make": by_make, "correlation": correlation}


def analyze_infrastructure(stations: pd.DataFrame) -> dict:
    """
    Summarise charging network composition.

    Returns a dict with:
        network_counts  – stations per network (top 10)
        connector_totals – {Level 1, Level 2, DC Fast} connector counts
        city_counts     – stations per city (top 15)
    """
    if not _require_columns(stations, ["network", "city"], "infrastructure"):
        return {}

    connector_cols = ["level1_count", "level2_count", "dc_fast_count"]
    if _require_columns(stations, connector_cols, "connectors"):
        stations = stations.copy()
        stations["total_connectors"] = stations[connector_cols].sum(axis=1)
        connector_totals = {
            "Level 1": int(stations["level1_count"].sum()),
            "Level 2": int(stations["level2_count"].sum()),
            "DC Fast": int(stations["dc_fast_count"].sum()),
        }
    else:
        connector_totals = {}

    return {
        "network_counts":   stations["network"].value_counts().head(10),
        "connector_totals": connector_totals,
        "city_counts":      stations["city"].value_counts().head(15),
        "stations":         stations,          # carries total_connectors if added
    }


def analyze_market_growth(sales: pd.DataFrame) -> dict:
    """
    Derive growth metrics from the sales time series.

    Returns a dict with:
        yearly_sales  – annual totals
        yoy_growth    – year-over-year % change
        monthly_avg   – mean sales by calendar month (seasonality)
        cagr          – compound annual growth rate (float)
    """
    if not _require_columns(sales, ["date", "total_ev_sales", "year", "month"], "market"):
        return {}

    yearly_sales = sales.groupby("year")["total_ev_sales"].sum()
    yoy_growth   = yearly_sales.pct_change() * 100

    n_years = len(yearly_sales) - 1
    cagr = (yearly_sales.iloc[-1] / yearly_sales.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        "yearly_sales": yearly_sales,
        "yoy_growth":   yoy_growth,
        "monthly_avg":  sales.groupby("month")["total_ev_sales"].mean(),
        "cagr":         cagr * 100,
    }


def analyze_manufacturer_trends(vehicles: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a linear trend to each manufacturer's efficiency over time.

    Returns a DataFrame with one row per manufacturer:
        Manufacturer, Average MPGe, Annual Improvement (MPGe),
        Annual Improvement (%), Year Range, Data Points, R-squared
    """
    if not _require_columns(vehicles, ["make", "year", "combined_mpge"], "mfr trends"):
        return pd.DataFrame()

    bev = _bev_only(vehicles)
    yearly_avg = bev.groupby(["make", "year"])["combined_mpge"].mean().reset_index()
    rows = []
    MIN_YEARS = 4    # fewer points make the linear trend statistically unreliable
    MIN_R2    = 0.3  # below this the trend is noise, not signal

    for make, grp in yearly_avg.groupby("make"):
        if len(grp) < MIN_YEARS:
            continue
        X = grp["year"].values.reshape(-1, 1)
        y = grp["combined_mpge"].values
        model = LinearRegression().fit(X, y)
        r2    = model.score(X, y)
        if r2 < MIN_R2:
            continue   # trend not reliable enough to report
        slope   = model.coef_[0]
        avg_eff = y.mean()
        rows.append({
            "Manufacturer":              make,
            "Average MPGe":              round(avg_eff, 1),
            "Annual Improvement (MPGe)": round(slope, 2),
            "Annual Improvement (%)":    round(slope / avg_eff * 100 if avg_eff else 0, 2),
            "Year Range":                f"{grp['year'].min()}–{grp['year'].max()}",
            "Data Points":               len(grp),
            "R-squared":                 round(r2, 3),
        })

    df = pd.DataFrame(rows).sort_values("Average MPGe", ascending=False).reset_index(drop=True)
    print(f"  [mfr trends] {len(df)} manufacturers passed ≥{MIN_YEARS} years + R²≥{MIN_R2} filter")
    return df


def segment_vehicles(vehicles: pd.DataFrame,
                     features: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """
    Segment BEVs into meaningful consumer market segments using
    rule-based quantile assignment.

    Why not GMM/K-Means:
      The EPA BEV dataset is a tight blob — 90% of vehicles sit in a
      narrow 90-130 MPGe / 250-350mi band. Statistical clustering splits
      this blob along arbitrary diagonal axes that produce segments where
      every cluster spans the full data range, making them useless for
      consumers. "Long Range cluster has lower avg range than Balanced
      cluster" is the symptom.

    Why rule-based quantiles work here:
      Consumer EV segments are defined by two independent axes:
        - Efficiency tier (MPGe): how far per kWh
        - Range tier (miles): how far per charge
      Cutting each axis at the 40th and 70th percentile produces a 3x3
      grid. We then merge the 9 cells into 4 meaningful named segments
      that are guaranteed to be monotonically ordered on both axes.

    Segments (assigned by quadrant):
      0 — City / Efficient  : high MPGe, lower range
      1 — Mainstream        : mid MPGe, mid range  (largest group)
      2 — Long Range        : higher range regardless of efficiency
      3 — Performance / SUV : lower MPGe, varies (trucks, performance)

    Returns (vehicles_with_cluster_column, n_segments).
    """
    if not _require_columns(vehicles, ["combined_mpge", "range_miles"], "segmentation"):
        return vehicles, 0

    bev = _bev_only(vehicles)
    if len(bev) < 20:
        print("  [segments] insufficient BEV rows — skipping.")
        return vehicles, 0

    # ── Quantile segmentation on deduplicated base models ───────────────────
    # Aggregate to base model first (one row per make+model), then apply
    # quantile cuts. This avoids the multi-year/trim contamination that
    # made previous approaches produce identical cluster averages.
    bev = bev.copy()
    bev["_make_k"]  = bev["make"].astype(str).str.strip().str.lower()
    bev["_model_k"] = bev["model"].apply(base_model_name)

    # Aggregate to make + base_model — average specs across all trims and years
    # Use max year so we know the model is current
    bev = bev.copy()
    bev["_mk"]   = bev["make"].astype(str).str.strip().str.lower()
    bev["_base"] = bev["model"].apply(base_model_name)

    # Aggregate to one row per base model (latest year, averaged specs)
    agg = (
        bev.groupby(["_mk", "_base"])
           .agg(mpge=("combined_mpge", "mean"),
                rng=("range_miles",   "mean"),
                latest=("year",        "max"))
           .reset_index()
    )
    # Only segment models that appear in 2022 or later (current market)
    agg = agg[agg["latest"] >= 2022].copy()

    # Quantile cuts on clean deduplicated current-market view
    mpge_lo = agg["mpge"].quantile(0.33)
    mpge_hi = agg["mpge"].quantile(0.67)
    rng_hi  = agg["rng"].quantile(0.67)

    def assign(row):
        e, r = row["mpge"], row["rng"]
        if e >= mpge_hi:   return 0   # High Efficiency
        if e < mpge_lo:    return 3   # Performance & SUV
        if r >= rng_hi:    return 2   # Long Range
        return 1                       # Mainstream

    agg["cluster"] = agg.apply(assign, axis=1)
    agg_lookup = {(r["_mk"], r["_base"]): r["cluster"] for _, r in agg.iterrows()}

    # Propagate back to all rows (all years, all trims of the same base model)
    def get_cluster(row):
        mk = str(row["make"]).strip().lower()
        base = base_model_name(row["model"])
        return agg_lookup.get((mk, base), np.nan)

    labels_all = bev.apply(get_cluster, axis=1).values
    valid  = ~np.isnan(labels_all)
    n_hit  = valid.sum()
    counts = np.bincount(labels_all[valid].astype(int), minlength=4)
    n      = 4

    print(f"  [segments] quantile on {len(agg)} base models | "
          f"cuts: MPGe {mpge_lo:.0f}/{mpge_hi:.0f}, Range —/{rng_hi:.0f} mi")
    print(f"  [segments] matched {n_hit}/{len(bev)} rows ({100*n_hit/len(bev):.0f}%)")
    for cid, name in [(0,"High Efficiency"),(1,"Mainstream"),
                      (2,"Long Range"),(3,"Performance & SUV")]:
        mask  = labels_all == cid
        avg_r = bev["range_miles"].values[mask].mean() if mask.any() else 0
        avg_e = bev["combined_mpge"].values[mask].mean() if mask.any() else 0
        print(f"    Seg {cid} {name:16s}: "
              f"MPGe={avg_e:.1f}  range={avg_r:.0f} mi  n={mask.sum()}")

    out = vehicles.copy()
    out["cluster"] = np.nan
    out.loc[bev.index, "cluster"] = labels_all
    # Null out pre-2022 rows — quantile cuts were calibrated on 2022+ models
    # so older rows may get wrong assignments
    if "year" in out.columns:
        out.loc[out["year"] < 2022, "cluster"] = np.nan
    out["cluster"] = out["cluster"].astype("Int64")
    out.drop(columns=[c for c in ["_mk","_base"] if c in out.columns],
             inplace=True, errors="ignore")

    # Build segment stats directly from the clean agg dataframe
    # so export_results doesn't need to recompute
    seg_stats = []
    for cid, name in [(0,"High Efficiency"),(1,"Mainstream"),
                      (2,"Long Range"),(3,"Performance & SUV")]:
        grp      = agg[agg["cluster"] == cid]
        full_bev = out[out["cluster"] == cid]
        if grp.empty:
            continue
        seg_stats.append({
            "cluster_id": cid,
            "count":      int(len(full_bev)),
            "avg_mpge":   round(float(grp["mpge"].mean()), 1),
            "avg_range":  round(float(grp["rng"].mean()),  1),
            "top_makes":  full_bev["make"].value_counts().head(3).index.tolist()
                          if "make" in full_bev.columns else [],
        })

    return out, n, seg_stats


def forecast_sales(sales: pd.DataFrame, periods: int = 12) -> dict:
    """
    Log-linear forecast of monthly EV sales.

    Returns a dict with:
        future_dates  – DatetimeIndex of forecast months
        forecast      – predicted sales (array)
        upper / lower – 95 % confidence interval arrays
        r2            – in-sample R²
    """
    if not _require_columns(sales, ["date", "total_ev_sales"], "forecast"):
        return {}

    df = sales.copy().sort_values("date").reset_index(drop=True)
    df["month_num"] = range(len(df))

    valid = df["total_ev_sales"] > 0
    X = df.loc[valid, "month_num"].values.reshape(-1, 1)
    y = np.log(df.loc[valid, "total_ev_sales"].values)

    model = LinearRegression().fit(X, y)
    log_std = (y - model.predict(X)).std()
    r2 = r2_score(np.exp(y), np.exp(model.predict(X)))

    last_month = df["month_num"].max()
    future_X   = np.arange(last_month + 1, last_month + periods + 1).reshape(-1, 1)
    log_pred   = model.predict(future_X)

    return {
        "future_dates": pd.date_range(
            start=df["date"].max() + pd.DateOffset(months=1),
            periods=periods, freq="ME",
        ),
        "forecast": np.exp(log_pred),
        "upper":    np.exp(log_pred + 1.96 * log_std),
        "lower":    np.exp(log_pred - 1.96 * log_std),
        "r2":       r2,
        "model":    model,
    }


# ── 4. Plot functions (consume analysis dicts, produce figures) ───────────────
def plot_efficiency(vehicles: pd.DataFrame, results: dict) -> None:
    """Four-panel efficiency overview."""
    _section("PLOTTING – Efficiency")
    if not results:
        return

    yearly, by_make, corr = results["yearly"], results["by_make"], results["correlation"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("EV Efficiency Analysis", fontsize=16, fontweight="bold")

    # Yearly trend with confidence band
    ax = axes[0, 0]
    ax.plot(yearly.index, yearly["mean"], marker="o", color=PALETTE[0], linewidth=2)
    ax.fill_between(yearly.index, yearly["mean"] - yearly["std"],
                    yearly["mean"] + yearly["std"], alpha=0.25, color=PALETTE[0])
    ax.set(title="Average Efficiency by Year", xlabel="Year", ylabel="Combined MPGe")
    ax.grid(True, alpha=0.3)

    # By manufacturer — horizontal bar so labels never overlap
    by_make.plot(kind="barh", ax=axes[0, 1], color=PALETTE[1])
    axes[0, 1].set(title="Efficiency by Manufacturer (BEV only)",
                   xlabel="Combined MPGe", ylabel="")
    axes[0, 1].invert_yaxis()
    axes[0, 1].bar_label(axes[0, 1].containers[0], fmt="%.0f", padding=3, fontsize=7)
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    # Range vs efficiency (coloured by year)
    sc1 = axes[1, 0].scatter(vehicles["combined_mpge"], vehicles["range_miles"],
                              c=vehicles["year"], cmap="viridis", alpha=0.7,
                              s=50, edgecolors="black", linewidth=0.4)
    axes[1, 0].set(title="Range vs Efficiency (by year)",
                   xlabel="Combined MPGe", ylabel="Range (miles)")
    plt.colorbar(sc1, ax=axes[1, 0], label="Year")
    axes[1, 0].grid(True, alpha=0.3)

    # Battery vs range — only plot if the column has real data
    batt_col = "battery_capacity_kwh"
    has_batt = batt_col in vehicles.columns and vehicles[batt_col].notna().any()
    if has_batt:
        sc2 = axes[1, 1].scatter(vehicles[batt_col], vehicles["range_miles"],
                                  c=vehicles["combined_mpge"], cmap="plasma", alpha=0.7,
                                  s=50, edgecolors="black", linewidth=0.4)
        axes[1, 1].set(title="Battery Size vs Range (by efficiency)",
                       xlabel="Battery Capacity (kWh)", ylabel="Range (miles)")
        plt.colorbar(sc2, ax=axes[1, 1], label="MPGe")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5,
                        "Battery capacity data not available.\nRun with --include-openev to enrich.",
                        ha="center", va="center", transform=axes[1, 1].transAxes,
                        fontsize=10, color="gray", style="italic")

    plt.tight_layout()
    _save_figure("ev_efficiency")
    plt.show()

    # Correlation heatmap (separate figure)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.8, cbar_kws={"shrink": 0.8})
    plt.title("Performance Metrics – Correlation Matrix", fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()
    _save_figure("ev_correlation")
    plt.show()


def plot_infrastructure(results: dict) -> None:
    """Four-panel charging infrastructure overview."""
    _section("PLOTTING – Infrastructure")
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Charging Infrastructure Analysis", fontsize=16, fontweight="bold")

    # Collapse networks under 3 % share into "Other" to keep labels readable,
    # then display as a horizontal bar rather than a crowded pie.
    nc = results["network_counts"].copy().astype(float)
    total = nc.sum()
    small_mask = nc / total < 0.03
    nc_clean = nc[~small_mask].copy()
    if small_mask.any():
        nc_clean["Other"] = nc[small_mask].sum()
    nc_pct = (nc_clean / total * 100).sort_values()

    nc_pct.plot(kind="barh", ax=axes[0, 0], color=PALETTE[0])
    axes[0, 0].set(title="Network Market Share", xlabel="Share (%)", ylabel="")
    axes[0, 0].bar_label(axes[0, 0].containers[0],
                         fmt="%.1f%%", padding=3, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    if results["connector_totals"]:
        axes[0, 1].bar(results["connector_totals"].keys(),
                       results["connector_totals"].values(), color=PALETTE[:3])
        axes[0, 1].set(title="Connector Types", ylabel="Total Connectors")

    results["city_counts"].plot(kind="barh", ax=axes[1, 0], color=PALETTE[3])
    axes[1, 0].set(title="Top 15 Cities by Station Count", xlabel="Stations")

    stations = results["stations"]
    if "access_code" in stations.columns:
        stations["access_code"].value_counts().plot(kind="pie", ax=axes[1, 1], autopct="%1.1f%%")
        axes[1, 1].set(title="Access Types", ylabel="")

    plt.tight_layout()
    _save_figure("ev_infrastructure")
    plt.show()


def plot_market_growth(sales: pd.DataFrame, results: dict) -> None:
    """Four-panel market growth overview."""
    _section("PLOTTING – Market Growth")
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("EV Market Growth Analysis", fontsize=16, fontweight="bold")

    # Monthly trend
    axes[0, 0].plot(sales["date"], sales["total_ev_sales"], color=PALETTE[0], linewidth=2)
    axes[0, 0].set(title="Monthly EV Sales", xlabel="Date", ylabel="Units Sold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=45)

    # YoY growth bars
    yoy = results["yoy_growth"].dropna()
    bars = axes[0, 1].bar(yoy.index, yoy.values, color=PALETTE[1])
    axes[0, 1].axhline(0, color="black", linewidth=0.8, alpha=0.4)
    axes[0, 1].set(title="Year-over-Year Growth", xlabel="Year", ylabel="Growth (%)")
    for bar in bars:
        h = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

    # Seasonality
    ma = results["monthly_avg"]
    axes[1, 0].plot(ma.index, ma.values, marker="o", color=PALETTE[2], linewidth=2)
    axes[1, 0].set(title="Seasonal Pattern", xlabel="Month", ylabel="Avg Units Sold")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)

    # Market share trend (if present)
    if "market_share_percent" in sales.columns:
        axes[1, 1].plot(sales["date"], sales["market_share_percent"],
                        color=PALETTE[3], linewidth=2)
        axes[1, 1].set(title="EV Market Share", xlabel="Date", ylabel="Market Share (%)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save_figure("ev_market_growth")
    plt.show()


def plot_manufacturer_trends(trends: pd.DataFrame, vehicles: pd.DataFrame) -> None:
    """
    Two-panel figure:
      Left  – actual yearly MPGe data points + regression trend line per manufacturer
      Right – horizontal bar of annual improvement rates
    """
    _section("PLOTTING – Manufacturer Trends")
    if trends.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Manufacturer Efficiency Trends", fontsize=16, fontweight="bold")

    # ── Left panel: trend lines ───────────────────────────────────────────────
    yearly_avg = (
        vehicles.groupby(["make", "year"])["combined_mpge"]
        .mean()
        .reset_index()
    )
    color_map = {make: PALETTE[i % len(PALETTE)]
                 for i, make in enumerate(trends["Manufacturer"])}

    for make in trends["Manufacturer"]:
        grp = yearly_avg[yearly_avg["make"] == make]
        if len(grp) < 2:
            continue
        color = color_map[make]
        years = grp["year"].values
        eff   = grp["combined_mpge"].values

        # Data points
        axes[0].scatter(years, eff, color=color, s=80, alpha=0.8,
                        edgecolors="black", linewidth=0.8, zorder=3)

        # Regression line extended ±0.5 yr
        model = LinearRegression().fit(years.reshape(-1, 1), eff)
        x_line = np.linspace(years.min() - 0.5, years.max() + 0.5, 100)
        axes[0].plot(x_line, model.predict(x_line.reshape(-1, 1)),
                     color=color, linewidth=2, alpha=0.7,
                     linestyle="--", label=make, zorder=2)

    axes[0].set(title="Yearly Efficiency with Trend Lines",
                xlabel="Year", ylabel="Combined MPGe")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Right panel: improvement rate bars ───────────────────────────────────
    sorted_df = trends.sort_values("Annual Improvement (%)")
    bar_colors = [PALETTE[3] if v < 0 else PALETTE[4]
                  for v in sorted_df["Annual Improvement (%)"]]
    axes[1].barh(sorted_df["Manufacturer"], sorted_df["Annual Improvement (%)"],
                 color=bar_colors, edgecolor="black", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=1.2)
    axes[1].set(title="Annual Improvement Rate (BEV, ≥4 yrs, R²≥0.3)",
                xlabel="Annual Improvement (%)")

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        v = row["Annual Improvement (%)"]
        offset = 0.05 if v >= 0 else -0.05
        axes[1].text(v + offset, i, f"{v:.2f}%",
                     va="center", ha="left" if v >= 0 else "right", fontsize=9)

    fig.text(0.5, -0.02,
             "Note: negative trends may reflect lineup expansion (new larger/heavier "
             "variants) rather than genuine efficiency regression.",
             ha="center", fontsize=8, color="gray", style="italic")
    plt.tight_layout()
    _save_figure("manufacturer_trends")
    plt.show()


def plot_forecast(sales: pd.DataFrame, fc: dict) -> None:
    """Sales forecast with confidence interval."""
    _section("PLOTTING – Sales Forecast")
    if not fc:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sales["date"], sales["total_ev_sales"],
            label="Historical", linewidth=2, color=PALETTE[0])
    ax.plot(fc["future_dates"], fc["forecast"],
            label="Forecast", linewidth=2, linestyle="--", color=PALETTE[1])
    ax.fill_between(fc["future_dates"], fc["lower"], fc["upper"],
                    color=PALETTE[1], alpha=0.2, label="95% CI")
    ax.text(0.02, 0.97, f"R² = {fc['r2']:.3f}", transform=ax.transAxes,
            va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set(title="EV Sales Forecast – Next 12 Months",
           xlabel="Date", ylabel="Monthly Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_figure("ev_forecast")
    plt.show()


# Fixed segment names matching rule-based assignment in segment_vehicles()
_SEG_NAMES  = {0: "High Efficiency", 1: "Mainstream", 2: "Long Range", 3: "Performance & SUV"}
_SEG_COLORS = ["#2E86AB", "#A23B72", "#06A77D", "#C73E1D"]


def plot_segmentation(vehicles: pd.DataFrame, k: int) -> None:
    """Single scatter — efficiency vs range coloured by consumer segment with cut lines."""
    _section("PLOTTING - Market Segmentation")
    if "cluster" not in vehicles.columns or k == 0:
        return

    bev = _bev_only(vehicles)
    bev = bev[bev["cluster"].notna() & (bev["year"] >= 2022)].copy()
    if bev.empty:
        print("  [segmentation] no clustered BEV rows.")
        return

    # Plot one dot per base model (latest year, 2022+)
    bev["_mk"] = bev["make"].astype(str).str.strip().str.lower()
    bev["_md"] = bev["model"].apply(base_model_name)
    bev = (bev.sort_values("year", ascending=False)
              .drop_duplicates(subset=["_mk", "_md"])
              .drop(columns=["_mk","_md"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("EV Market Segments — Consumer Profiles (current models)",
                 fontsize=14, fontweight="bold")

    for cid in sorted(bev["cluster"].dropna().unique()):
        mask  = bev["cluster"] == cid
        label = _SEG_NAMES.get(int(cid), f"Segment {cid}")
        color = _SEG_COLORS[int(cid) % len(_SEG_COLORS)]
        ax.scatter(bev.loc[mask, "combined_mpge"],
                   bev.loc[mask, "range_miles"],
                   c=color, label=f"{label} (n={mask.sum()})",
                   alpha=0.7, s=50, edgecolors="black", linewidth=0.3)

    ax.set(xlabel="Efficiency (MPGe)", ylabel="Range (miles)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure("ev_segmentation")
    plt.show()

# ── 5. Reporting ──────────────────────────────────────────────────────────────
def print_summary(data: EVDataset,
                  growth: dict,
                  mfr_trends: pd.DataFrame,
                  k: int) -> None:
    """Print a concise executive summary to stdout."""
    _section("EXECUTIVE SUMMARY")
    v, s, sa = data.vehicles, data.stations, data.sales

    print(f"  Report date  : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Vehicles     : {len(v):,}  |  Stations : {len(s):,}  |  Sales months : {len(sa)}")
    print(f"  Manufacturers: {v['make'].nunique()}  |  Market CAGR : {growth.get('cagr', 0):.1f}%")

    if not mfr_trends.empty:
        leader = mfr_trends.iloc[0]
        print(f"  Efficiency leader : {leader['Manufacturer']} ({leader['Average MPGe']:.1f} MPGe)")
        avg_imp = mfr_trends["Annual Improvement (%)"].mean()
        print(f"  Industry avg improvement : {avg_imp:.2f}% / year")

    if "cluster" in data.vehicles.columns:
        print(f"  Market segments  : {k}")

    print()


def export_results(data: EVDataset, seg_stats: list | None = None) -> None:
    """Write processed CSVs and a JSON metadata file to PROCESSED_DIR."""
    _section("EXPORTING RESULTS")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")

    paths = {
        "vehicles": PROCESSED_DIR / f"epa_vehicles_{stamp}.csv",
        "stations": PROCESSED_DIR / f"charging_stations_{stamp}.csv",
        "sales":    PROCESSED_DIR / f"ev_sales_{stamp}.csv",
    }
    data.vehicles.to_csv(paths["vehicles"], index=False)
    data.stations.to_csv(paths["stations"], index=False)
    data.sales.to_csv(paths["sales"],    index=False)

    meta = {
        "export_date": stamp,
        "datasets": {
            name: {"rows": len(getattr(data, name)), "columns": len(getattr(data, name).columns)}
            for name in ["vehicles", "stations", "sales"]
        },
    }
    meta_path = PROCESSED_DIR / f"metadata_{stamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    for name, path in {**paths, "metadata": meta_path}.items():
        print(f"  ✓ {name:10s} → {path}")

    # ── Pre-computed segment stats — written by segment_vehicles ─────────────
    if seg_stats:
        seg_path = PROCESSED_DIR / "segment_stats.json"
        seg_path.write_text(json.dumps(seg_stats, indent=2))
        print(f"  ✓ segment_stats → {seg_path}")
        for s in seg_stats:
            print(f"    Seg {s['cluster_id']}: "
                  f"MPGe={s['avg_mpge']}  range={s['avg_range']} mi  n={s['count']}")


# ── 6. Pipeline orchestrator ──────────────────────────────────────────────────
def run_full_pipeline(data: EVDataset) -> None:
    """
    Run every analysis and plot in order.

    Steps
    -----
    1. Prepare / clean sales data
    2. Analyse efficiency, infrastructure, growth, manufacturer trends
    3. Segment vehicles with Gaussian Mixture Model (BIC model selection)
    4. Forecast next 12 months of sales
    5. Plot everything
    6. Print executive summary
    7. Export processed data
    """
    # ── Preparation ──────────────────────────────────────────────────────────
    data.sales = prepare_sales(data.sales)

    # ── Analysis ─────────────────────────────────────────────────────────────
    eff_results  = analyze_efficiency(data.vehicles)
    infra        = analyze_infrastructure(data.stations)
    growth       = analyze_market_growth(data.sales)
    mfr_trends   = analyze_manufacturer_trends(data.vehicles)
    data.vehicles, k, seg_stats_cache = segment_vehicles(data.vehicles)
    fc           = forecast_sales(data.sales)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_efficiency(data.vehicles, eff_results)
    plot_infrastructure(infra)
    plot_market_growth(data.sales, growth)
    plot_manufacturer_trends(mfr_trends, data.vehicles)
    plot_forecast(data.sales, fc)
    plot_segmentation(data.vehicles, k)

    # ── Summary & export ──────────────────────────────────────────────────────
    print_summary(data, growth, mfr_trends, k)
    export_results(data, seg_stats_cache)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = load_datasets(_DEFAULT_RAW_DIR)
    run_full_pipeline(dataset)