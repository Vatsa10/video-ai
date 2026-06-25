# HF Spaces (sdk: docker) — live FastAPI + WebSocket UI on port 7860.
# Real-time YOLOE detection wants a GPU; on CPU it runs but is slow.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as a non-root user (uid 1000). Create it + writable dirs.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app
COPY --chown=user requirements.txt .
# CPU torch by default. For a GPU Space, install a CUDA torch wheel instead.
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# storage/ (shards, frames, uploads) must be writable at runtime
RUN mkdir -p storage && chown -R user:user /app /home/user
USER user

EXPOSE 7860
CMD ["uvicorn", "videoqa.live.main:app", "--host", "0.0.0.0", "--port", "7860", "--ws", "wsproto"]
