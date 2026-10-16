#!/usr/bin/env bash
set -e

HOST="${QDRANT_HOST:-qdrant_db}"
PORT="${QDRANT_PORT:-6333}"

echo "[SemanticEdge] Waiting for Qdrant at http://${HOST}:${PORT}/healthz ..."
until wget -qO- "http://${HOST}:${PORT}/healthz" > /dev/null 2>&1; do
  echo "[SemanticEdge] Qdrant not ready yet, retrying in 2 seconds..."
  sleep 2
done

echo "[SemanticEdge] Qdrant is up!"

if [ ! -f "models/siglip_int8/vision_encoder_int8.onnx" ] || [ ! -f "models/siglip_int8/text_encoder_int8.onnx" ]; then
  echo "[SemanticEdge] SigLIP ONNX models missing. Auto-exporting split INT8 models (one-time)..."
  python tools/export_split_models.py
fi

echo "[SemanticEdge] Launching cloud backend service..."
exec "$@"
