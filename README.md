# EV Explorer

A consumer-facing electric vehicle intelligence tool built on real government data. Browse, filter, and compare every current EV on the US market — with charging station coverage, market segmentation, and manufacturer efficiency trends.

**Live:** [ev-explorer.up.railway.app](https://ev-explorer.up.railway.app)

---

## Features

- **Vehicle browser** — 1,200+ EPA-certified EVs grouped by manufacturer, with trim-level expand/collapse and AI-powered model name normalization
- **Side-by-side comparison** — select any two vehicles to compare range, efficiency, battery capacity, and charging speed head-to-head
- **Annual fuel cost** — estimated electricity cost and savings vs. gas on every vehicle card, computed from EPA MPGe ratings
- **Charging map** — 80,000+ US stations from NREL AFDC, loaded by viewport bounds as you pan and zoom
- **Market segments** — four buyer profiles (High Efficiency, Mainstream, Long Range, Performance & SUV) derived from quantile analysis on deduplicated base-model averages
- **Efficiency trends** — manufacturer MPGe trajectories from 2019–2027, filterable by brand

---

## Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python · pandas · NumPy |
| API | FastAPI · uvicorn |
| Frontend | React · Vite · Plotly.js · Leaflet |
| AI normalization | Anthropic Claude API (batch, cached) |
| Deployment | Docker · Railway |

---

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| Vehicle fuel economy | EPA fueleconomy.gov bulk export | 1,631 vehicles · 89 columns |
| Charging stations | NREL Alternative Fuels Station API | 80,203 US stations |
| Battery & charging specs | OpenEV v1.24.0 | 229 matched vehicles |

---

## Architecture

```
data/raw/                        ← EPA CSV + NREL stations (downloaded)
notebooks/ev_analysis.py         ← Analysis pipeline
  ├── BEV filtering              (fuel_type, is_phev, MPGe ≥ 50)
  ├── Quantile segmentation      (on deduplicated base-model averages)
  ├── Manufacturer trend fitting (OLS, R² ≥ 0.3 quality gate)
  └── Export → outputs/processed_data/

ev-app/
  ├── backend/main.py            ← FastAPI · pre-computed JSON · lru_cache
  └── frontend/src/
        ├── pages/               ← Compare · Chargers · Segments · Trends
        └── components/
              └── MakeGrid.jsx   ← AI model normalization + trim grouping
```

**Key design decisions:**
- Pre-computed segment stats at pipeline time — backend serves JSON, no live clustering
- Viewport-bound station fetching — Leaflet `moveend` triggers bbox API call, not radius search
- AI normalization runs once at backend startup in parallel batches of 80, cached to disk
- Quantile cuts applied to deduplicated base-model averages (not raw trim rows) to avoid contamination

---

## Running Locally

```bash
# 1. Clone and install
git clone https://github.com/evanfu/ev-performance-analysis
cd ev-performance-analysis

# 2. Fetch data
pip install -r requirements.txt
cp .env.example .env          # add your NREL_API_KEY and ANTHROPIC_API_KEY
python src/data_collection.py

# 3. Run pipeline
python notebooks/ev_analysis.py

# 4. Start backend
cd ev-app/backend
uvicorn main:app --reload --port 8000

# 5. Start frontend (new terminal)
cd ev-app/frontend
npm install
npm run dev
```

---

## Notes

- Negative manufacturer efficiency trends reflect lineup expansion (larger/heavier variants) rather than genuine regression — noted in the UI
- Segment assignments are computed on 2022+ vehicles only; older model years inherit the same segment label but are excluded from segment statistics
- Battery and charging specs cover ~230 vehicles (Tesla, Hyundai, BMW, Audi, Volkswagen, Rivian have best coverage)