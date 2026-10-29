import pytest


def test_viewport_filters_before_count_and_pagination(client):
    result = client.get(
        "/api/stations", params=dict(lat_min=33, lat_max=35, lon_min=-119, lon_max=-117)
    ).json()
    assert result["total"] == 1
    assert [s["station_id"] for s in result["results"]] == [1]


def test_viewport_supports_antimeridian(client):
    result = client.get("/api/stations?lat_min=50&lat_max=52&lon_min=178&lon_max=-178").json()
    assert result["total"] == 2


@pytest.mark.parametrize(
    "query",
    [
        "lat_min=33",
        "lat_min=40&lat_max=30&lon_min=0&lon_max=1",
        "lat_min=-91&lat_max=10&lon_min=0&lon_max=1",
        "limit=0",
        "offset=-1",
    ],
)
def test_invalid_station_queries_are_rejected(client, query):
    assert client.get("/api/stations?" + query).status_code == 422


def test_literal_network_filter_and_fast_charger_filter(client):
    result = client.get("/api/stations?network=A%20(US)&dc_fast_only=true&limit=1").json()
    assert result["total"] == 2 and len(result["results"]) == 1


def test_nearby_distance_and_coordinates(client):
    result = client.get("/api/stations/nearby?lat=34&lon=-118&radius_km=1").json()
    assert len(result) == 1 and result[0]["distance_km"] == pytest.approx(0)
    assert client.get("/api/stations/nearby?lat=91&lon=0").status_code == 422


def test_vehicle_detail_uses_source_id_and_handles_missing(client):
    result = client.get("/api/vehicles?limit=1").json()["results"][0]
    assert result["id"] >= 1000
    assert client.get(f"/api/vehicles/{result['id']}").json() == result
    assert client.get("/api/vehicles/-1").status_code == 404


@pytest.mark.parametrize("query", ["limit=-1", "offset=-1", "sort_by=garbage", "cluster=4"])
def test_invalid_vehicle_queries_are_rejected(client, query):
    assert client.get("/api/vehicles?" + query).status_code == 422


def test_null_values_are_json_null(client):
    assert client.get("/api/vehicles?limit=1").json()["results"][0]["msrp_base"] is None


def test_vehicle_filter_and_pagination(client):
    result = client.get("/api/vehicles?min_range=300&limit=2&offset=1").json()
    assert result["total"] > 2 and len(result["results"]) == 2
    assert all(v["range_miles"] >= 300 for v in result["results"])
