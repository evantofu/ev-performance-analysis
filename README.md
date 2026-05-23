# EV Explorer

Consumer-facing EV comparison and charging station finder.
Real EPA vehicle data + NREL charging station data + GMM market segmentation.

## Stack

- **Backend**: FastAPI (Python) — reads pre-processed CSVs, serves JSON
- **Frontend**: React + Vite + Plotly + Three.js
- **Deployment**: Docker (single container)

## Quick start

### 1. Generate the data

From your project root:

```bash
# Collect vehicle + station data
python data_collection.py --nrel-key YOUR_KEY

# Optional: enrich with battery/price/charging specs
python data_collection.py --nrel-key YOUR_KEY --include-openev

# Run analysis pipeline (produces outputs/processed_data/*.csv)
python notebooks/ev_analysis.py
```

### 2. Run with Docker

```bash
cd ev-app
docker compose up
```

Open http://localhost:8000

### 3. Local development (hot reload)

Terminal 1 — FastAPI backend:
```bash
cd ev-app
pip install -r backend/requirements.txt
cd backend
PROCESSED_DIR=../../outputs/processed_data RAW_DIR=../../data/raw \
  uvicorn main:app --reload --port 8000
```

Terminal 2 — React frontend:
```bash
cd ev-app/frontend
npm install --legacy-peer-deps
npm run dev
```

Open http://localhost:5173

### 4. Deploy to Railway / Render

1. Push the `ev-app/` directory to a GitHub repo
2. Connect to Railway or Render, set build command to `docker build`
3. Mount your data as a volume at `/app/data` and `/app/outputs`
4. Set environment variable `NREL_API_KEY` if running data collection in CI

## API reference

| Endpoint                    | Description                              |
|-----------------------------|------------------------------------------|
| `GET /api/summary`          | Top-level KPIs                           |
| `GET /api/vehicles`         | Filtered vehicle list (paginated)        |
| `GET /api/vehicles/{id}`    | Single vehicle detail                    |
| `GET /api/vehicles/makes`   | All manufacturers with counts            |
| `GET /api/stations`         | Filtered station list                    |
| `GET /api/stations/nearby`  | Stations within radius of lat/lon        |
| `GET /api/stations/networks`| Network breakdown                        |
| `GET /api/segments`         | GMM cluster summaries                    |
| `GET /api/trends`           | Manufacturer yearly MPGe averages        |

## Features

- **Compare EVs** — filter by make, range, price, year. Interactive scatter.
  Select up to 2 vehicles for side-by-side metric comparison.
- **Find Chargers** — Leaflet map of 80k+ US stations. Filter by network,
  state, DC Fast only. Colour-coded by charger speed.
- **Market Segments** — GMM clustering with BIC model selection.
  Toggle between 2D Plotly scatter and 3D Three.js orbital explorer.
- **Trends** — Manufacturer efficiency trends 2019–present. BEV-only filter.

## Notes

- Sales forecast charts are intentionally excluded from the consumer app
  (the source data has simulated components — not appropriate to show as fact).
- The 3D segment view shows efficiency × range × price. Price axis requires
  running `--include-openev` during data collection.
- The Leaflet map loads from CDN on first use (~50KB, cached after).
