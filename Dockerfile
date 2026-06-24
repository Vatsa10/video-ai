# HF Spaces (sdk: docker) — runs the live FastAPI + WebSocket UI on port 7860.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU torch by default; for a GPU Space, swap to a CUDA wheel index.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV HF_HOME=/app/.cache
EXPOSE 7860
CMD ["uvicorn", "videoqa.live.main:app", "--host", "0.0.0.0", "--port", "7860", "--ws", "wsproto"]
