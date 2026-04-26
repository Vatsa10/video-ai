#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
pip install -e analysis
pip install -r backend/requirements.txt
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
