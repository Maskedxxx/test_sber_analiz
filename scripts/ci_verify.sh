#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Building Docker image..."
docker compose build

echo "[2/3] Running live GigaChat E2E tests inside container..."
docker compose run --rm \
  -e REBUILD_COLLECTION=true \
  chatbot \
  python tests/test_gigachat_live.py

echo "[3/3] Done. See test_results/gigachat_live.json for details."

