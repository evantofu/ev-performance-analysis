import numpy as np
import pandas as pd
import pytest
from src.vehicle_rules import base_model_name, consumer_bevs, manufacturer_trends, segment_vehicles


def test_filtering_preserves_index_and_segment_alignment(vehicles):
    excluded = vehicles.iloc[[0]].copy()
    excluded.index = [3]
    excluded["fuel_type"] = "Gasoline"
    excluded["is_phev"] = True
    source = pd.concat([excluded, vehicles])
    out, _, _ = segment_vehicles(source)
    assert pd.isna(out.loc[3, "cluster"])
    assert out.loc[vehicles.index, "cluster"].notna().all()
    pd.testing.assert_index_equal(out.index, source.index)


@pytest.mark.parametrize("mode", ["small", "missing", "old", "null"])
def test_insufficient_data_has_consistent_return_contract(vehicles, mode):
    if mode == "small":
        vehicles = vehicles.head(5)
    elif mode == "missing":
        vehicles = vehicles.drop(columns="range_miles")
    elif mode == "old":
        vehicles["year"] = 2020
    else:
        vehicles["range_miles"] = np.nan
    out, count, stats = segment_vehicles(vehicles)
    assert count == 0 and stats == []
    assert out["cluster"].isna().all()


def test_segmentation_uses_latest_year_specs(vehicles):
    old = vehicles.copy()
    old["year"] = 2022
    old["combined_mpge"] = 51
    out, _, stats = segment_vehicles(pd.concat([old, vehicles], ignore_index=True))
    assert out["cluster"].notna().all()
    assert min(s["avg_mpge"] for s in stats) > 51
    assert sum(s["base_model_count"] for s in stats) == 25
    assert sum(s["count"] for s in stats) == 50


def test_old_or_invalid_rows_are_never_labeled(vehicles):
    vehicles.loc[vehicles.index[0], "year"] = 2021
    vehicles.loc[vehicles.index[1], "range_miles"] = np.nan
    out, _, _ = segment_vehicles(vehicles)
    assert out.iloc[:2]["cluster"].isna().all()
    assert out.iloc[2:]["cluster"].notna().all()


def test_missing_fuel_identity_does_not_imply_bev(vehicles):
    assert consumer_bevs(vehicles.drop(columns="fuel_type")).empty


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Model 3 Long Range AWD", "model 3"),
        ("e-tron GT", "e-tron gt"),
        ("Unlisted Car AWD", "unlisted car"),
        ("AWD", "awd"),
    ],
)
def test_normalization_preserves_model_identity(raw, expected):
    assert base_model_name(raw) == expected


def test_manufacturer_trends_enforce_both_gates():
    rows = []
    for make, values in [
        ("Linear", [80, 90, 100, 110]),
        ("Short", [80, 90, 100]),
        ("Flat", [90, 90, 90, 90]),
    ]:
        rows.extend(
            dict(make=make, year=2020 + i, combined_mpge=value, fuel_type="Electricity")
            for i, value in enumerate(values)
        )
    yearly, fits = manufacturer_trends(pd.DataFrame(rows))
    assert set(yearly["make"]) == {"Linear"}
    assert fits.iloc[0]["r2"] == pytest.approx(1)
