import json
import pandas as pd
import ev_analysis as analysis


def test_missing_sales_skips_forecasts_unless_explicitly_enabled(tmp_path, vehicles, stations):
    vehicles.to_csv(tmp_path / "epa_vehicles_20260101.csv", index=False)
    stations.to_csv(tmp_path / "charging_stations_20260101.csv", index=False)
    data = analysis.load_datasets(str(tmp_path))
    assert data.sales.empty and data.sales_source == "not_available"
    assert analysis.forecast_sales(data.sales) == {}
    demo = analysis.load_datasets(str(tmp_path), allow_synthetic_sales=True)
    assert demo.sales_source == "synthetic_demo" and not demo.sales.empty
    assert not list(tmp_path.glob("ev_sales_data_*.csv"))


def test_export_retains_sales_provenance(tmp_path, monkeypatch, vehicles, stations):
    monkeypatch.setattr(analysis, "PROCESSED_DIR", tmp_path)
    data = analysis.EVDataset(vehicles, stations, pd.DataFrame(), sales_source="synthetic_demo")
    analysis.export_results(data)
    metadata = json.loads(next(tmp_path.glob("metadata_*.json")).read_text())
    assert metadata["provenance"]["sales"] == "synthetic_demo"


def test_no_manufacturer_passes_is_an_empty_result(vehicles):
    assert analysis.analyze_manufacturer_trends(vehicles).empty


def test_forecast_reports_chronological_holdout_against_baseline():
    import numpy as np

    sales = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=48, freq="MS"),
            "total_ev_sales": np.exp(np.arange(48) * 0.01) * 1000,
        }
    )
    result = analysis.forecast_sales(sales)
    assert len(result["forecast"]) == 12
    assert result["holdout_mae"] < 1e-6
    assert result["naive_mae"] > 0


def test_forecast_skips_nonpositive_observations():
    sales = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=48, freq="MS"), "total_ev_sales": [0] * 48}
    )
    assert analysis.forecast_sales(sales) == {}
