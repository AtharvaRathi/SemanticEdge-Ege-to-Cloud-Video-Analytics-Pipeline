"""
db/setup_collection.py
======================
Creates (or re-creates) the Qdrant collection with:

  1. Scalar Quantization (INT8)
       - Compresses stored vectors from FP32 → INT8 (≈4× RAM reduction)
       - quantile=0.99 preserves accuracy by calibrating the quantization
         range on the top 99 % of the vector value distribution.
       - always_ram=False → quantized vectors live on disk (mmap), not RAM.

  2. HNSW index tuned for low-resource hardware
       - m=16, ef_construct=100 (safe defaults; lower m saves RAM/disk)

  3. Payload indexes for fast multi-tenant filtering (user_id, video_id)

Usage
-----
    # Ensure Qdrant is reachable, then:
    python db/setup_collection.py
    python db/setup_collection.py --recreate   # drop & rebuild

Environment variables (override via .env or shell):
    QDRANT_HOST            (default: localhost)
    QDRANT_PORT            (default: 6333)
    QDRANT_COLLECTION      (default: video_embeddings)
    SIGLIP_EMBEDDING_DIM   (default: 768  – siglip-base-patch16-224)
"""

import argparse
import logging
import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("setup_collection")

# ---------------------------------------------------------------------------
# Config (read from env to stay consistent with app/core/config.py)
# ---------------------------------------------------------------------------

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "video_embeddings")

# SigLIP base produces 768-dimensional embeddings.
EMBEDDING_DIM = int(os.getenv("SIGLIP_EMBEDDING_DIM", 768))


# ---------------------------------------------------------------------------
# Setup logic
# ---------------------------------------------------------------------------

def setup_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the Qdrant collection with INT8 scalar quantization."""

    # ── Optional: drop existing collection ─────────────────────────────
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            log.warning("--recreate flag set. Dropping collection '%s'…", COLLECTION_NAME)
            client.delete_collection(COLLECTION_NAME)
            log.info("Collection '%s' deleted.", COLLECTION_NAME)
        else:
            log.info(
                "Collection '%s' already exists. "
                "Use --recreate to drop and rebuild it.",
                COLLECTION_NAME,
            )
            _ensure_payload_indexes(client)
            return

    # ── Create collection with quantization ────────────────────────────
    log.info(
        "Creating collection '%s' (dim=%d, INT8 scalar quantization)…",
        COLLECTION_NAME,
        EMBEDDING_DIM,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,

        # ── Vector space ───────────────────────────────────────────────
        vectors_config=qmodels.VectorParams(
            size=EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
            # Raw FP32 vectors on disk — only read during rescore (few reads).
            on_disk=True,
        ),

        # ── Scalar Quantization (INT8) ─────────────────────────────────
        # Converts FP32 → INT8: 4× smaller (768 B vs 3 072 B per vector).
        # always_ram=True: INT8 index stays HOT in RAM → every ANN search
        # is a pure-memory operation → fastest possible query latency.
        # quantile=0.99 calibrates the [min, max] range on the top 99th
        # percentile, preventing outliers from squashing precision.
        quantization_config=qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,   # ← KEY: INT8 index in RAM for speed
            )
        ),

        # ── HNSW index ─────────────────────────────────────────────────
        # m=16: 16 bi-directional edges per node — good recall/RAM tradeoff.
        # ef_construct=200: higher quality graph — better recall at query time.
        # on_disk defaults to False → graph stays in RAM → fast traversal.
        hnsw_config=qmodels.HnswConfigDiff(
            m=16,
            ef_construct=200,
            # on_disk=False (default) — keep HNSW graph in RAM
        ),

        # Payload on disk — read only when returning final results (tiny reads).
        on_disk_payload=True,
    )

    log.info("Collection '%s' created successfully.", COLLECTION_NAME)
    _ensure_payload_indexes(client)


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Create keyword indexes on user_id and video_id fields for fast
    multi-tenant filtering without full-collection scans.
    """
    for field in ("user_id", "video_id"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            log.info("Payload index on '%s' ensured.", field)
        except Exception as exc:
            # Index may already exist; that's fine.
            log.debug("Index on '%s' skipped (%s).", field, exc)


def verify_collection(client: QdrantClient) -> None:
    """Print a summary of the created collection's configuration."""
    info = client.get_collection(COLLECTION_NAME)
    log.info("=" * 55)
    log.info("  Collection '%s' verified", COLLECTION_NAME)
    log.info("  Vectors dim      : %s", info.config.params.vectors.size)
    log.info("  Distance metric  : %s", info.config.params.vectors.distance)
    log.info(
        "  Quantization     : %s",
        info.config.quantization_config or "none",
    )
    log.info("  Points count     : %s", info.points_count)
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap Qdrant collection with INT8 scalar quantization."
    )
    parser.add_argument(
        "--host", default=QDRANT_HOST, help="Qdrant host (default from env)"
    )
    parser.add_argument(
        "--port", type=int, default=QDRANT_PORT, help="Qdrant port (default from env)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection before creating a new one",
    )
    args = parser.parse_args()

    log.info("Connecting to Qdrant at %s:%d…", args.host, args.port)
    try:
        client = QdrantClient(host=args.host, port=args.port, timeout=30)
        # Quick connectivity check
        client.get_collections()
    except Exception as exc:
        log.error("Cannot connect to Qdrant: %s", exc)
        log.error("Make sure Qdrant is running (e.g. docker compose up qdrant_db)")
        sys.exit(1)

    log.info("Connected.")
    setup_collection(client, recreate=args.recreate)
    verify_collection(client)


if __name__ == "__main__":
    main()
