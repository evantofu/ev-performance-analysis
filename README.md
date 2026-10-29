# EV Explorer

Compare EPA-listed electric vehicles by range, efficiency, and operating cost, and explore US charging-station locations on a map.

**[Hosted demo](https://ev-explorer.up.railway.app)** · **[CI](https://github.com/evantofu/ev-performance-analysis/actions)** · Python / FastAPI / React / Vite / pandas / Plotly / Leaflet

![EV Explorer vehicle comparison, with filters, efficiency chart, and operating-cost estimates](docs/ev-explorer.png)

## What you can do

- Browse **1,274 consumer BEV model-year/trim records**, covering 2019–2027 in the bundled snapshot. Expand manufacturers and models, filter the list, and compare two records.
- Compare range, MPGe, available battery/charging specifications, and EPA annual fuel costs. Missing specifications stay missing.
- See electricity-cost estimates with explicit assumptions: 13,500 miles/year, $0.16/kWh, and a gasoline comparison at $3.50/gallon and 25 mpg.
- Explore **80,203 charging-station locations**. Panning and zooming send geographic bounds to the API; filters apply before counting and pagination.
- Inspect four descriptive efficiency/range groups and manufacturer efficiency trends that pass a minimum four-year history and an in-sample R² threshold of 0.3.

## Run locally — no API keys required

Requires **Python 3.12** and **Node.js 22.12+**. Bundled data is included; no downloads or paid services are needed for the application.

```bash
git clone https://github.com/evantofu/ev-performance-analysis.git
cd ev-performance-analysis
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r ev-app/backend/requirements.txt
python -m uvicorn main:app --app-dir ev-app/backend --reload --port 8000
```

In a second terminal, from the repository root:

```bash
cd ev-app/frontend
npm ci
npm run dev
```

Open **http://localhost:5173**. The Vite development server proxies API requests to port 8000. Interactive API documentation is at **http://localhost:8000/docs**. `/health` checks the process; `/ready` also loads and validates dataset availability.

To serve the production frontend locally, run `npm run build` in `ev-app/frontend` and restart the backend. Open **http://localhost:8000**; direct links such as `/compare` are supported.

### Docker

```bash
docker compose up --build
# Application: http://localhost:8000
# Optional frontend development server: docker compose --profile dev up --build
```

The image builds the frontend and includes the shared Python rules and processed snapshot. `PORT` defaults to 8000. To use another dataset directory or additional browser origins, export `PROCESSED_DIR` or comma-separated `CORS_ORIGINS` before starting the API. Vite uses `API_PROXY_TARGET` for its server-side proxy and optional `VITE_API_URL` for an explicit browser API origin.

## Data and method

The bundled source snapshot is dated **May 22, 2026**. These are historical records and station locations, not a live catalog or real-time charger availability feed.

| Input | Source | Bundled scope |
|---|---|---|
| Vehicle records | [EPA fuel-economy bulk export](https://www.fueleconomy.gov/feg/download.shtml) | 1,631 source records; 1,274 pass this application's consumer-BEV cohort filters |
| Station locations | [AFDC API](https://developer.nlr.gov/docs/transportation/alt-fuel-stations-v1/) | 80,203 locations |
| Additional specifications | [OpenEV](https://github.com/open-ev-data/open-ev-data-dataset) | Partial coverage; missing values are retained |

**Identity and grouping.** EPA vehicle IDs remain stable when filtering or rebuilding. Model grouping uses a bundled name lookup with a deterministic regex fallback, shared by the pipeline and API. It does not call a live AI service. Grouping is heuristic and can merge variants; the original EPA model name remains visible.

**Segments.** For the 2022+ cohort, use each base model's latest available model year and average its trims. Apply efficiency cuts at the 33rd and 67th percentiles, then an upper-range cut at the 67th percentile within the remaining middle-efficiency group. Labels are High Efficiency, Mainstream, Longer Range, and Lower Efficiency. Eligible older records of the same base model inherit its label; pre-2022 and invalid rows remain unassigned. This is a descriptive rule system, not trained clustering, a body-style classifier, or a purchasing recommendation. Segment averages are base-model-weighted; displayed record counts include trims and years.

**Trends.** Yearly manufacturer averages are trim-weighted. OLS R² measures in-sample fit, not predictive accuracy. Changing vehicle lineups can change averages without showing a technical improvement or regression in the same vehicle.

**Limitations.** The 50 MPGe floor is a cohort choice, not a universal definition of a BEV. OpenEV enrichment can use cross-year or fuzzy matches and should be checked against manufacturer specifications before a purchase. Station results are capped at 2,000 in the UI; zoom in to narrow the viewport. Map tiles and address search require internet access. MSRP coverage is incomplete. Operating-cost estimates exclude charging losses, public charging premiums, taxes, and maintenance.

## Architecture

| Layer | Responsibility |
|---|---|
| `src/data_collection.py` | Download EPA/AFDC data and optionally enrich from OpenEV; timeout/retry handling |
| `src/vehicle_rules.py` | Shared filtering, model identity, index-safe segmentation, and trend gates |
| `notebooks/ev_analysis.py` | Offline analysis, figures, optional illustrative sales extrapolation, and metadata exports |
| `outputs/processed_data/` | Bundled snapshot and provenance; source dates remain explicit |
| `ev-app/backend/main.py` | Read-only FastAPI routes, cached DataFrames, validated filters, vectorized nearby search, and SPA hosting |
| `ev-app/frontend/src/` | React pages, shared components, Zustand selection, abortable requests, and visible error states |

This read-heavy snapshot fits in memory. A database is unnecessary at this scale; each API worker caches its own DataFrames. Restart the API after replacing snapshot files. Chart and map pages load on demand; Plotly uses the smaller basic chart bundle.

Example bounded request:

```bash
curl 'http://localhost:8000/api/stations?lat_min=33&lat_max=35&lon_min=-119&lon_max=-117&limit=10'
```

All four bounds must be supplied together. Longitude bounds can cross the antimeridian. Limits and offsets are validated, and response `total` counts the filtered rows before pagination. Network filters use literal text, not regular expressions.

## Refresh and analyze data

The quick start uses the bundled snapshot. To collect fresh inputs, install the additional pipeline dependencies and obtain an AFDC key:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
# Set NREL_API_KEY in .env; the collector loads it.
python src/data_collection.py --include-openev
MPLBACKEND=Agg python notebooks/ev_analysis.py
```

Collection writes `data/raw/`; analysis writes dated processed data and figures. Source APIs and schemas may change. Review the output and restart the API after a refresh.

To rebuild bundled segment labels and metadata without downloading data:

```bash
python scripts/rebuild_snapshot.py
```

Sales are not used by the web application. Missing sales data skips sales charts. `--synthetic-sales` explicitly enables labeled demo sales for offline analysis; it never saves those inputs as real observations. User-supplied sales are marked unverified. Illustrative log-linear extrapolation reports a final-12-month holdout MAE against a last-value baseline; residual bands are not calibrated forecast intervals. No claim of forecasting skill is made from synthetic data.

## Verify changes

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
MPLBACKEND=Agg python -m pytest -q

cd ev-app/frontend
npm ci
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Python tests cover row alignment, excluded/invalid vehicles, return contracts, trend gates, stable IDs, geographic bounds, pagination, and provenance. Frontend tests cover cancellation, stale responses, API errors, null segment labels, and cost assumptions. Browser tests exercise the production build, comparisons, map-bound requests, deep links, and visible failures. GitHub Actions runs the same checks.
