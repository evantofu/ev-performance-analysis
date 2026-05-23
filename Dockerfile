# ── Stage 1: Build React ──────────────────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY ev-app/frontend/package.json ev-app/frontend/package-lock.json* ./
RUN npm install --legacy-peer-deps
COPY ev-app/frontend/ ./
RUN npm run build

# ── Stage 2: Python backend ───────────────────────────────────────────────────
FROM python:3.12-slim AS backend
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY ev-app/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ev-app/backend/ ./
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
COPY outputs/processed_data/ ./outputs/processed_data/

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Use shell form so uvicorn runs from /app where main.py lives
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2