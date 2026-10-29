import pandas as pd
import pytest
from fastapi.testclient import TestClient
import main


@pytest.fixture
def vehicles():
    rows = [
        dict(
            make="Test",
            model=f"Model {i}",
            year=2025,
            fuel_type="Electricity",
            is_phev=False,
            combined_mpge=80 + i * 2,
            range_miles=200 + (i % 7) * 30,
            fueleconomy_id=1000 + i,
            battery_capacity_kwh=60,
            msrp_base=None,
        )
        for i in range(25)
    ]
    return pd.DataFrame(rows, index=range(10, 60, 2))


@pytest.fixture
def stations():
    return pd.DataFrame(
        [
            dict(
                station_id=1,
                station_name="<script>bad</script>",
                city="LA",
                state="CA",
                network="A (US)",
                latitude=34.0,
                longitude=-118.0,
                dc_fast_count=2,
                level1_count=0,
                level2_count=4,
            ),
            dict(
                station_id=2,
                station_name="NY",
                city="NY",
                state="NY",
                network="B",
                latitude=41.0,
                longitude=-74.0,
                dc_fast_count=0,
                level1_count=0,
                level2_count=2,
            ),
            dict(
                station_id=3,
                station_name="East",
                city="Island",
                state="AK",
                network="A (US)",
                latitude=51.0,
                longitude=179.0,
                dc_fast_count=1,
                level1_count=0,
                level2_count=0,
            ),
            dict(
                station_id=4,
                station_name="West",
                city="Island",
                state="AK",
                network="B",
                latitude=51.0,
                longitude=-179.0,
                dc_fast_count=0,
                level1_count=0,
                level2_count=1,
            ),
        ]
    )


@pytest.fixture
def client(monkeypatch, vehicles, stations):
    from src.vehicle_rules import base_model_name, segment_vehicles

    df, _, _ = segment_vehicles(vehicles)
    df["id"] = df["fueleconomy_id"]
    df["base_model"] = df["model"].map(base_model_name)
    monkeypatch.setattr(main, "_load_vehicles", lambda: df)
    monkeypatch.setattr(main, "_load_stations", lambda: stations)
    with TestClient(main.app) as test_client:
        yield test_client
