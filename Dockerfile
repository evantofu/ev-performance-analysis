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

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Python deps
COPY ev-app/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY ev-app/backend/ ./

# Copy built React app — FastAPI will serve it as static files
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copy processed data
COPY outputs/processed_data/ ./outputs/processed_data/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]