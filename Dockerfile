FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
COPY ev-app/frontend/package.json ev-app/frontend/package-lock.json ./
RUN npm ci
COPY ev-app/frontend/ ./
RUN npm run build

FROM python:3.12-slim AS backend
WORKDIR /app
COPY ev-app/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ev-app/backend/main.py ./main.py
COPY src/ ./src/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
COPY outputs/processed_data/ ./outputs/processed_data/
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.getenv('PORT','8000')+'/ready')"
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
