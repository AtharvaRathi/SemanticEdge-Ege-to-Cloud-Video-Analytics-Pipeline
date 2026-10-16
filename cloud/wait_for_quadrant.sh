#!/usr/bin/env bash
set -e

HOST="${QDRANT_HOST:-qdrant_db}"
PORT="${QDRANT_PORT:-6333}"

echo "[SemanticEdge] Waiting for Qdrant at http://${HOST}:${PORT}/healthz ..."
until wget -qO- "http://${HOST}:${PORT}/healthz" > /dev/null 2>&1; do
  echo "[SemanticEdge] Qdrant not ready yet, retrying in 2 seconds..."
  sleep 2
done

echo "[SemanticEdge] Qdrant is up! Launching cloud backend service..."
exec "$@"
