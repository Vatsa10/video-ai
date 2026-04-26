FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Project layout (repo root):
#   /app/pyproject.toml         analysis package config
#   /app/analysis/              Python package
#   /app/backend/               FastAPI service
COPY pyproject.toml /app/pyproject.toml
COPY analysis /app/analysis
RUN pip install --no-cache-dir -e /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend

ENV VIDEO_AI_STORAGE=/app/backend/storage
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
