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
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#06A77D", "#8B4513"]

# Anchor all paths to the project root (parent of whichever directory this
# file lives in — works whether run from project root, notebooks/, or anywhere).
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "notebooks" else _HERE
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_data"
_DEFAULT_RAW_DIR = str(PROJECT_ROOT / "data" / "raw")
sys.path.insert(0, str(PROJECT_ROOT))
from src.vehicle_rules import (  # noqa: E402
    base_model_name,
    consumer_bevs as _bev_only,
    segment_vehicles,
    manufacturer_trends,
)

plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 10})
sns.set_style("whitegrid")


# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class EVDataset:
    """Holds the three core DataFrames throughout the pipeline."""

    vehicles: pd.DataFrame
    stations: pd.DataFrame
    sales: pd.DataFrame
    sales_source: str = "unknown"


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


# ── 1. Data loading ───────────────────────────────────────────────────────────
def _make_synthetic_sales() -> pd.DataFrame:
    """
    Generate a plausible monthly US EV sales time series (2019–present).
    Available only with --synthetic-sales; outputs retain explicit provenance.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01", periods=(datetime.now().year - 2019 + 1) * 12, freq="MS")
    base = 5_500
    rows = []
    for i, d in enumerate(dates):
        trend = base * (1.055 ** (i / 12))  # ~5.5 % annual growth
        seasonal = 1 + 0.12 * np.sin((d.month - 3) * np.pi / 6)
        noise = rng.normal(1, 0.04)
        sales = max(0, int(trend * seasonal * noise))
        rows.append(
            {
                "date": d,
                "year": d.year,
                "month": d.month,
                "total_ev_sales": sales,
                "market_share_percent": round(min(sales / 700_000 * 100, 15), 2),
            }
        )
    df = pd.DataFrame(rows)
    # Save so subsequent runs reload instead of regenerating
    return df


def load_datasets(raw_dir: str = "data/raw", *, allow_synthetic_sales: bool = False) -> EVDataset:
    """
    Load vehicles and stations CSVs from *raw_dir*.

    Sales data is optional. Synthetic sales requires explicit opt-in; missing
    sales otherwise skips market-growth and forecasting charts.
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
        df = pd.read_csv(path, low_memory=False)
        frames[name] = df
        print(f"  {name:10s}: {len(df):,} rows × {len(df.columns)} cols  ← {path}")

    sales_matches = sorted(glob.glob(os.path.join(raw_dir, "ev_sales_data_*.csv")))
    source = "not_available"
    if sales_matches:
        frames["sales"] = pd.read_csv(sales_matches[-1])
        # Unverified user inputs are not asserted to be real observations.
        source = "user_supplied_unverified"
    elif allow_synthetic_sales:
        frames["sales"] = _make_synthetic_sales()
        source = "synthetic_demo"
    else:
        frames["sales"] = pd.DataFrame(columns=["date", "year", "month", "total_ev_sales"])
    return EVDataset(**frames, sales_source=source)


# ── 2. Data preparation ───────────────────────────────────────────────────────
def prepare_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, add year/month columns, drop the incomplete current year."""
    df = sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
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

    yearly = bev.groupby("year")["combined_mpge"].agg(["mean", "std"])
    # Require ≥5 total EPA-tested rows per manufacturer.
    # This excludes niche/exotic brands (Bugatti Rimac, Lordstown) that have
    # minimal US market presence and distort the efficiency scale.
    trim_counts = bev.groupby("make").size()
    qualified = trim_counts[trim_counts >= 5].index
    by_make = (
        bev[bev["make"].isin(qualified)].groupby("make")["combined_mpge"].mean().sort_values(ascending=False)
    )
    vehicles = bev  # use filtered set for correlation too

    num_cols = [
        c
        for c in ["combined_mpge", "range_miles", "battery_capacity_kwh", "year", "msrp_base"]
        if c in vehicles.columns
    ]
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
        "network_counts": stations["network"].value_counts().head(10),
        "connector_totals": connector_totals,
        "city_counts": stations["city"].value_counts().head(15),
        "stations": stations,  # carries total_connectors if added
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
    if sales.empty or not _require_columns(sales, ["date", "total_ev_sales", "year", "month"], "market"):
        return {}

    yearly_sales = sales.groupby("year")["total_ev_sales"].sum()
    yoy_growth = yearly_sales.pct_change() * 100

    n_years = len(yearly_sales) - 1
    cagr = (yearly_sales.iloc[-1] / yearly_sales.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        "yearly_sales": yearly_sales,
        "yoy_growth": yoy_growth,
        "monthly_avg": sales.groupby("month")["total_ev_sales"].mean(),
        "cagr": cagr * 100,
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

    yearly, fits = manufacturer_trends(vehicles)
    rows = []
    for fit in fits.to_dict("records"):
        grp = yearly.loc[yearly["make"].eq(fit["make"])]
        average = grp["combined_mpge"].mean()
        rows.append(
            {
                "Manufacturer": fit["make"],
                "Average MPGe": round(average, 1),
                "Annual Improvement (MPGe)": round(fit["slope"], 2),
                "Annual Improvement (%)": round(fit["slope"] / average * 100, 2),
                "Year Range": f"{grp['year'].min()}–{grp['year'].max()}",
                "Data Points": fit["years"],
                "R-squared": round(fit["r2"], 3),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("Average MPGe", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def forecast_sales(sales: pd.DataFrame, periods: int = 12) -> dict:
    """
    Log-linear forecast of monthly EV sales.

    Returns a dict with:
        future_dates  – DatetimeIndex of forecast months
        forecast      – predicted sales (array)
        upper / lower – approximate residual bands, not calibrated intervals
        r2            – in-sample R²
    """
    if len(sales) < 24 or not _require_columns(sales, ["date", "total_ev_sales"], "forecast"):
        return {}

    df = sales.copy().sort_values("date").reset_index(drop=True)
    df["month_num"] = range(len(df))

    valid = df["total_ev_sales"] > 0
    X = df.loc[valid, "month_num"].values.reshape(-1, 1)
    y = np.log(df.loc[valid, "total_ev_sales"].values)

    if len(y) < 24:
        return {}
    # Evaluate the final 12 observed months before refitting on all observations.
    split = len(y) - 12
    holdout_model = LinearRegression().fit(X[:split], y[:split])
    observed = np.exp(y[split:])
    holdout_mae = float(np.abs(observed - np.exp(holdout_model.predict(X[split:]))).mean())
    naive_mae = float(np.abs(observed - np.exp(y[split - 1])).mean())
    model = LinearRegression().fit(X, y)
    log_std = (y - model.predict(X)).std()
    r2 = r2_score(np.exp(y), np.exp(model.predict(X)))

    last_month = df["month_num"].max()
    future_X = np.arange(last_month + 1, last_month + periods + 1).reshape(-1, 1)
    log_pred = model.predict(future_X)

    return {
        "future_dates": pd.date_range(
            start=df["date"].max() + pd.DateOffset(months=1),
            periods=periods,
            freq="ME",
        ),
        "forecast": np.exp(log_pred),
        "upper": np.exp(log_pred + 1.96 * log_std),
        "lower": np.exp(log_pred - 1.96 * log_std),
        "r2": r2,
        "holdout_mae": holdout_mae,
        "naive_mae": naive_mae,
        "model": model,
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
    ax.fill_between(
        yearly.index,
        yearly["mean"] - yearly["std"],
        yearly["mean"] + yearly["std"],
        alpha=0.25,
        color=PALETTE[0],
    )
    ax.set(title="Average Efficiency by Year", xlabel="Year", ylabel="Combined MPGe")
    ax.grid(True, alpha=0.3)

    # By manufacturer — horizontal bar so labels never overlap
    by_make.plot(kind="barh", ax=axes[0, 1], color=PALETTE[1])
    axes[0, 1].set(title="Efficiency by Manufacturer (BEV only)", xlabel="Combined MPGe", ylabel="")
    axes[0, 1].invert_yaxis()
    axes[0, 1].bar_label(axes[0, 1].containers[0], fmt="%.0f", padding=3, fontsize=7)
    axes[0, 1].grid(True, alpha=0.3, axis="x")

    # Range vs efficiency (coloured by year)
    sc1 = axes[1, 0].scatter(
        vehicles["combined_mpge"],
        vehicles["range_miles"],
        c=vehicles["year"],
        cmap="viridis",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.4,
    )
    axes[1, 0].set(title="Range vs Efficiency (by year)", xlabel="Combined MPGe", ylabel="Range (miles)")
    plt.colorbar(sc1, ax=axes[1, 0], label="Year")
    axes[1, 0].grid(True, alpha=0.3)

    # Battery vs range — only plot if the column has real data
    batt_col = "battery_capacity_kwh"
    has_batt = batt_col in vehicles.columns and vehicles[batt_col].notna().any()
    if has_batt:
        sc2 = axes[1, 1].scatter(
            vehicles[batt_col],
            vehicles["range_miles"],
            c=vehicles["combined_mpge"],
            cmap="plasma",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.4,
        )
        axes[1, 1].set(
            title="Battery Size vs Range (by efficiency)",
            xlabel="Battery Capacity (kWh)",
            ylabel="Range (miles)",
        )
        plt.colorbar(sc2, ax=axes[1, 1], label="MPGe")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.5,
            "Battery capacity data not available.\nRun with --include-openev to enrich.",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=10,
            color="gray",
            style="italic",
        )

    plt.tight_layout()
    _save_figure("ev_efficiency")
    plt.show()

    # Correlation heatmap (separate figure)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
    )
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
    axes[0, 0].bar_label(axes[0, 0].containers[0], fmt="%.1f%%", padding=3, fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis="x")

    if results["connector_totals"]:
        axes[0, 1].bar(
            results["connector_totals"].keys(), results["connector_totals"].values(), color=PALETTE[:3]
        )
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


def plot_market_growth(sales: pd.DataFrame, results: dict, source: str = "unknown") -> None:
    """Four-panel market growth overview."""
    _section("PLOTTING – Market Growth")
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"EV Market Growth — {source}", fontsize=16, fontweight="bold")

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
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2, h, f"{h:.0f}%", ha="center", va="bottom", fontsize=9
        )

    # Seasonality
    ma = results["monthly_avg"]
    axes[1, 0].plot(ma.index, ma.values, marker="o", color=PALETTE[2], linewidth=2)
    axes[1, 0].set(title="Seasonal Pattern", xlabel="Month", ylabel="Avg Units Sold")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)

    # Market share trend (if present)
    if "market_share_percent" in sales.columns:
        axes[1, 1].plot(sales["date"], sales["market_share_percent"], color=PALETTE[3], linewidth=2)
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
    yearly_avg = manufacturer_trends(vehicles)[0]
    color_map = {make: PALETTE[i % len(PALETTE)] for i, make in enumerate(trends["Manufacturer"])}

    for make in trends["Manufacturer"]:
        grp = yearly_avg[yearly_avg["make"] == make]
        if len(grp) < 2:
            continue
        color = color_map[make]
        years = grp["year"].values
        eff = grp["combined_mpge"].values

        # Data points
        axes[0].scatter(years, eff, color=color, s=80, alpha=0.8, edgecolors="black", linewidth=0.8, zorder=3)

        # Regression line extended ±0.5 yr
        model = LinearRegression().fit(years.reshape(-1, 1), eff)
        x_line = np.linspace(years.min() - 0.5, years.max() + 0.5, 100)
        axes[0].plot(
            x_line,
            model.predict(x_line.reshape(-1, 1)),
            color=color,
            linewidth=2,
            alpha=0.7,
            linestyle="--",
            label=make,
            zorder=2,
        )

    axes[0].set(title="Yearly Efficiency with Trend Lines", xlabel="Year", ylabel="Combined MPGe")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Right panel: improvement rate bars ───────────────────────────────────
    sorted_df = trends.sort_values("Annual Improvement (%)")
    bar_colors = [PALETTE[3] if v < 0 else PALETTE[4] for v in sorted_df["Annual Improvement (%)"]]
    axes[1].barh(
        sorted_df["Manufacturer"],
        sorted_df["Annual Improvement (%)"],
        color=bar_colors,
        edgecolor="black",
        linewidth=1,
    )
    axes[1].axvline(0, color="black", linewidth=1.2)
    axes[1].set(title="Annual Improvement Rate (BEV, ≥4 yrs, R²≥0.3)", xlabel="Annual Improvement (%)")

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        v = row["Annual Improvement (%)"]
        offset = 0.05 if v >= 0 else -0.05
        axes[1].text(v + offset, i, f"{v:.2f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)

    fig.text(
        0.5,
        -0.02,
        "Note: negative trends may reflect lineup expansion (new larger/heavier "
        "variants) rather than genuine efficiency regression.",
        ha="center",
        fontsize=8,
        color="gray",
        style="italic",
    )
    plt.tight_layout()
    _save_figure("manufacturer_trends")
    plt.show()


def plot_forecast(sales: pd.DataFrame, fc: dict, source: str = "unknown") -> None:
    """Illustrative extrapolation with uncalibrated residual bands."""
    _section("PLOTTING – Sales Forecast")
    if not fc:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sales["date"], sales["total_ev_sales"], label="Historical", linewidth=2, color=PALETTE[0])
    ax.plot(
        fc["future_dates"], fc["forecast"], label="Forecast", linewidth=2, linestyle="--", color=PALETTE[1]
    )
    ax.fill_between(
        fc["future_dates"],
        fc["lower"],
        fc["upper"],
        color=PALETTE[1],
        alpha=0.2,
        label="Approx. residual band (not calibrated forecast interval)",
    )
    ax.text(
        0.02,
        0.97,
        f"Training R² = {fc['r2']:.3f}\n12-month holdout MAE: {fc['holdout_mae']:,.0f}; naive: {fc['naive_mae']:,.0f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set(title=f"Illustrative Sales Extrapolation — {source}", xlabel="Date", ylabel="Monthly Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_figure("ev_forecast")
    plt.show()


# Fixed segment names matching rule-based assignment in segment_vehicles()
_SEG_NAMES = {0: "High Efficiency", 1: "Mainstream", 2: "Longer Range", 3: "Lower Efficiency"}
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
    latest = bev.groupby(["_mk", "_md"])["year"].transform("max")
    bev = (
        bev.loc[bev["year"].eq(latest)]
        .groupby(["_mk", "_md"])
        .agg(
            combined_mpge=("combined_mpge", "mean"),
            range_miles=("range_miles", "mean"),
            cluster=("cluster", "first"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("EV Segments — Latest-Year Base-Model Means (2022+ cohort)", fontsize=14, fontweight="bold")

    for cid in sorted(bev["cluster"].dropna().unique()):
        mask = bev["cluster"] == cid
        label = _SEG_NAMES.get(int(cid), f"Segment {cid}")
        color = _SEG_COLORS[int(cid) % len(_SEG_COLORS)]
        ax.scatter(
            bev.loc[mask, "combined_mpge"],
            bev.loc[mask, "range_miles"],
            c=color,
            label=f"{label} (n={mask.sum()})",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.3,
        )

    ax.set(xlabel="Efficiency (MPGe)", ylabel="Range (miles)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure("ev_segmentation")
    plt.show()


# ── 5. Reporting ──────────────────────────────────────────────────────────────
def print_summary(data: EVDataset, growth: dict, mfr_trends: pd.DataFrame, k: int) -> None:
    """Print a concise executive summary to stdout."""
    _section("EXECUTIVE SUMMARY")
    v, s, sa = data.vehicles, data.stations, data.sales

    print(f"  Report date  : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Vehicles     : {len(v):,}  |  Stations : {len(s):,}  |  Sales months : {len(sa)}")
    cagr = f"{growth['cagr']:.1f}%" if growth else "not available"
    print(f"  Manufacturers: {v['make'].nunique()}  |  Market CAGR : {cagr}")

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
        "sales": PROCESSED_DIR / f"ev_sales_{stamp}.csv",
    }
    data.vehicles.to_csv(paths["vehicles"], index=False)
    data.stations.to_csv(paths["stations"], index=False)
    data.sales.to_csv(paths["sales"], index=False)

    meta = {
        "export_date": stamp,
        "provenance": {"sales": data.sales_source, "vehicles": "EPA + optional OpenEV", "stations": "AFDC"},
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
            print(
                f"    Seg {s['cluster_id']}: MPGe={s['avg_mpge']}  range={s['avg_range']} mi  n={s['count']}"
            )


# ── 6. Pipeline orchestrator ──────────────────────────────────────────────────
def run_full_pipeline(data: EVDataset) -> None:
    """
    Run every analysis and plot in order.

    Steps
    -----
    1. Prepare / clean sales data
    2. Analyse efficiency, infrastructure, growth, manufacturer trends
    3. Assign descriptive quantile segments to consumer BEVs
    4. Forecast next 12 months of sales
    5. Plot everything
    6. Print executive summary
    7. Export processed data
    """
    # ── Preparation ──────────────────────────────────────────────────────────
    data.sales = prepare_sales(data.sales)

    # ── Analysis ─────────────────────────────────────────────────────────────
    eff_results = analyze_efficiency(data.vehicles)
    infra = analyze_infrastructure(data.stations)
    growth = analyze_market_growth(data.sales)
    mfr_trends = analyze_manufacturer_trends(data.vehicles)
    data.vehicles, k, seg_stats_cache = segment_vehicles(data.vehicles)
    fc = forecast_sales(data.sales)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_efficiency(data.vehicles, eff_results)
    plot_infrastructure(infra)
    if not data.sales.empty:
        plot_market_growth(data.sales, growth, source=data.sales_source)
    plot_manufacturer_trends(mfr_trends, data.vehicles)
    if data.sales_source == "synthetic_demo":
        print("  Sales and forecasts are SYNTHETIC DEMO DATA.")
    plot_forecast(data.sales, fc, source=data.sales_source)
    plot_segmentation(data.vehicles, k)

    # ── Summary & export ──────────────────────────────────────────────────────
    print_summary(data, growth, mfr_trends, k)
    export_results(data, seg_stats_cache)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build EV analysis outputs")
    parser.add_argument("--raw-dir", default=_DEFAULT_RAW_DIR)
    parser.add_argument(
        "--synthetic-sales", action="store_true", help="Explicitly opt into labeled demo sales"
    )
    args = parser.parse_args()
    dataset = load_datasets(args.raw_dir, allow_synthetic_sales=args.synthetic_sales)
    run_full_pipeline(dataset)
