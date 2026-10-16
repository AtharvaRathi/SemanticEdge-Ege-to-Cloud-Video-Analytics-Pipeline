"""
app/services/qdrant.py
======================
Qdrant client wrapper — Speed-first, RAM-aware configuration.

Connection strategy
-------------------
1. LOCAL EMBEDDED (default for dev): QdrantClient(path=...) — zero network,
   zero Docker dependency, fastest possible I/O on Windows/Linux/Mac.
2. HTTP SERVER (production / Docker): set QDRANT_USE_SERVER=true in .env
   or environment to connect to host:port instead.

Storage strategy (speed first, then RAM)
-----------------------------------------
* HNSW index          → RAM  (fast ANN graph traversal, ~small)
* INT8 quantized vecs → RAM  (always_ram=True) — this is what the graph
                              actually searches; keeping it in RAM is the
                              single biggest speed lever.
* Raw FP32 vectors    → disk (on_disk=True) — only accessed during rescore,
                              which reads ≤2× limit vectors, not the full set.
* Payload / metadata  → disk — rarely read at query time.

This gives sub-10 ms search latency while keeping the RAM footprint low
because the INT8 index is 4× smaller than FP32 (768 B vs 3 072 B/vector).
"""

from __future__ import annotations


import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from app.core.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _make_client() -> QdrantClient:
    """
    Build a QdrantClient using the best available mode:

      QDRANT_USE_SERVER=true  →  HTTP client to QDRANT_HOST:QDRANT_PORT
      (default)               →  Embedded local client (no Docker needed)

    The embedded client stores data in ``qdrant_local_db/`` at the project
    root.  It is binary-compatible with the HTTP server, so you can snapshot
    and import the same data into a Qdrant Docker container for production.
    """
    if settings.QDRANT_USE_SERVER:
        host = settings.QDRANT_HOST
        port = settings.QDRANT_PORT
        log.info(
            "Qdrant mode: HTTP server at %s:%d  (set QDRANT_USE_SERVER=false for embedded)",
            host, port,
        )
        return QdrantClient(
            host=host,
            port=port,
            timeout=60,
            # Uncomment for ~2× throughput on bulk upsert via gRPC:
            # grpc_port=6334,
            # prefer_grpc=True,
        )
    else:
        local_path = settings.QDRANT_LOCAL_PATH
        os.makedirs(local_path, exist_ok=True)
        log.info(
            "Qdrant mode: LOCAL EMBEDDED at '%s'  (set QDRANT_USE_SERVER=true for HTTP)",
            local_path,
        )
        return QdrantClient(path=local_path)


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class QdrantService:

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION
        self.client: QdrantClient | None = None
        self._connect()

    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Establish client connection with a clear error if it fails."""
        try:
            self.client = _make_client()
            # Quick connectivity probe
            self.client.get_collections()
            log.info("Qdrant connection OK.")
        except Exception as exc:
            log.error(
                "Cannot connect to Qdrant: %s\n"
                "  → For local dev:  make sure QDRANT_USE_SERVER is NOT set (uses embedded mode)\n"
                "  → For Docker:     run  docker-compose up -d qdrant_db  then set QDRANT_USE_SERVER=true",
                exc,
            )
            self.client = None
            # Don't raise — the app still starts; errors surface at request time.

    # ------------------------------------------------------------------
    def _ensure_client(self) -> None:
        """Reconnect if the client was never initialised."""
        if self.client is None:
            log.warning("Qdrant client not connected — retrying…")
            self._connect()
        if self.client is None:
            raise RuntimeError(
                "Qdrant is unavailable.  Check logs for connection details."
            )

    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """
        Create the collection with speed-first settings if it doesn't exist.

        What lives in RAM vs disk
        -------------------------
        RAM:  HNSW graph + INT8 quantized index   ← searched on every query
        Disk: Raw FP32 vectors + payload            ← read only for rescore/scroll
        """
        self._ensure_client()
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            log.info("Collection '%s' exists.", self.collection_name)
            return

        log.info("Creating collection '%s' (speed-first config)…", self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,

            # ── Vector space ───────────────────────────────────────────
            vectors_config=qm.VectorParams(
                size=768,
                distance=qm.Distance.COSINE,
                # Raw FP32 vectors on disk → only read during rescore (small reads)
                on_disk=True,
            ),

            # ── INT8 Scalar Quantization ───────────────────────────────
            # 4× smaller than FP32 (768 B vs 3 072 B per vector).
            # always_ram=True: keep the INT8 index HOT in RAM so every ANN
            # search is a pure-memory operation → fastest possible latency.
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,   # ← KEY: INT8 index in RAM for speed
                )
            ),

            # ── HNSW index ─────────────────────────────────────────────
            # m=16 gives good recall; ef_construct=200 builds a higher-quality
            # graph (costs more at index time, pays off at search time).
            # on_disk=False: HNSW graph stays in RAM for fast traversal.
            hnsw_config=qm.HnswConfigDiff(
                m=16,
                ef_construct=200,
                # on_disk defaults to False → graph in RAM ← FAST
            ),

            # Payload on disk — only accessed when returning results (small reads)
            on_disk_payload=True,
        )

        # Keyword indexes for O(log n) multi-tenant filtering
        for field in ("user_id", "video_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # already exists

        log.info("Collection '%s' created.", self.collection_name)

    # ------------------------------------------------------------------
    def upsert_vectors(
        self,
        user_id: str,
        video_id: str,
        vectors: list,
        timestamps: list,
        filename: str = "",
        frame_indices: list | None = None,
        metadata_list: list | None = None,
    ) -> int:
        """
        Upsert embedding vectors with metadata.
        Batches in groups of 128 for optimal network / memory balance.

        Parameters
        ----------
        filename      : original video filename (for human-readable search results)
        frame_indices : sequential frame positions matching each vector/timestamp
        metadata_list : optional list of metadata dicts per vector
        """
        self._ensure_client()
        self._ensure_collection()

        if frame_indices is None:
            frame_indices = list(range(len(vectors)))

        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "user_id": user_id,
                    "video_id": video_id,
                    "timestamp": timestamps[i],
                    "frame_idx": frame_indices[i],
                    "filename": filename,
                    **(metadata_list[i] if metadata_list and i < len(metadata_list) and isinstance(metadata_list[i], dict) else {}),
                },
            )
            for i, vec in enumerate(vectors)
        ]

        batch_size = 128
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

        log.info(
            "Upserted %d vectors (user='%s', video='%s', file='%s').",
            len(points), user_id, video_id, filename,
        )
        return len(points)

    # ------------------------------------------------------------------
    def search(self, user_id: str, query_vector: list, limit: int = 5) -> list:
        """
        ANN search filtered by user_id.

        Search path
        -----------
        1. HNSW graph traversal on INT8 vectors (both in RAM → fast)
        2. Oversampling: retrieve 2× limit candidates
        3. Rescore top candidates against raw FP32 vectors (disk reads, few)
        4. Return top `limit` by true cosine score

        Uses query_points() — the current API for qdrant-client >= 1.10.
        The old client.search() was removed in that release.
        """
        self._ensure_client()

        query_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="user_id",
                    match=qm.MatchValue(value=user_id),
                )
            ]
        )

        search_params = qm.SearchParams(
            hnsw_ef=128,   # search-time expansion factor; higher = better recall
            exact=False,   # ANN (fast); set True only for ground-truth debugging
            quantization=qm.QuantizationSearchParams(
                ignore=False,
                rescore=True,       # Re-rank top candidates with FP32 vectors
                oversampling=2.0,
            ),
        )

        try:
            # query_points() is the unified search API in qdrant-client >= 1.10.
            # It returns a QueryResponse; the scored hits are in .points.
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,          # NOTE: 'query=', not 'query_vector='
                query_filter=query_filter,
                search_params=search_params,
                limit=limit,
                with_payload=True,
            )
            results = response.points        # list[ScoredPoint] — .score, .payload intact
            log.info(
                "Search user='%s' → %d results (query_dim=%d).",
                user_id, len(results), len(query_vector),
            )
            return results

        except Exception as exc:
            log.error("Qdrant search failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    def health_check(self) -> dict:
        """Return basic connectivity and collection info for /health endpoint."""
        try:
            self._ensure_client()
            collections = [c.name for c in self.client.get_collections().collections]
            count = 0
            if self.collection_name in collections:
                info = self.client.get_collection(self.collection_name)
                count = info.points_count or 0
            return {
                "qdrant": "ok",
                "mode": "embedded" if not os.getenv("QDRANT_USE_SERVER") else "server",
                "collection": self.collection_name,
                "points": count,
            }
        except Exception as exc:
            return {"qdrant": "error", "detail": str(exc)}

    # ------------------------------------------------------------------
    def delete_video(self, user_id: str, video_id: str) -> dict:
        """
        Delete all embedding vectors for a specific video belonging to a user.

        Uses server-side FilterSelector so only matching points are removed —
        no client-side fetch needed.  Safe to call even if the video doesn't exist.
        """
        self._ensure_client()

        # Count before delete so we can report back how many frames were removed
        count_before = self._count_for_filter(
            qm.Filter(must=[
                qm.FieldCondition(key="user_id",  match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id)),
            ])
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[
                    qm.FieldCondition(key="user_id",  match=qm.MatchValue(value=user_id)),
                    qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id)),
                ])
            ),
        )

        log.info(
            "Deleted video_id='%s' for user='%s' (%d frames removed).",
            video_id, user_id, count_before,
        )
        return {"deleted": True, "video_id": video_id, "frames_removed": count_before}

    # ------------------------------------------------------------------
    def list_user_videos(self, user_id: str) -> list:
        """
        Return a list of unique videos for a user, with frame counts.
        Scrolls all matching points and aggregates in Python.

        Returns [{video_id, filename, frame_count}, …] sorted by filename.
        """
        self._ensure_client()
        video_map: dict = {}
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qm.Filter(must=[
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))
                ]),
                with_payload=["video_id", "filename"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in results:
                vid_id   = point.payload.get("video_id", "")
                filename = point.payload.get("filename", "")
                if vid_id not in video_map:
                    video_map[vid_id] = {
                        "video_id":    vid_id,
                        "filename":    filename,
                        "frame_count": 0,
                    }
                video_map[vid_id]["frame_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return sorted(video_map.values(), key=lambda x: x["filename"])

    # ------------------------------------------------------------------
    def list_users(self) -> list:
        """
        Admin: return every unique user_id found in the collection with stats.
        Scrolls the entire collection and aggregates by user_id.

        Returns [{user_id, video_count, total_frames}, …] sorted by user_id.
        NOTE: For very large collections (millions of vectors) consider adding
        a dedicated users metadata collection instead of scanning here.
        """
        self._ensure_client()
        user_map: dict = {}
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["user_id", "video_id"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in results:
                uid    = point.payload.get("user_id",  "") or ""
                vid_id = point.payload.get("video_id", "") or ""
                if uid not in user_map:
                    user_map[uid] = {"user_id": uid, "_video_ids": set(), "total_frames": 0}
                user_map[uid]["_video_ids"].add(vid_id)
                user_map[uid]["total_frames"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return sorted(
            [
                {
                    "user_id":      uid,
                    "video_count":  len(data["_video_ids"]),
                    "total_frames": data["total_frames"],
                }
                for uid, data in user_map.items()
            ],
            key=lambda x: x["user_id"],
        )

    # ------------------------------------------------------------------
    def delete_user(self, user_id: str) -> dict:
        """
        Admin: delete ALL data belonging to a user — every embedding for
        every video they ever uploaded.

        Returns a summary of what was removed.
        """
        self._ensure_client()

        # Gather stats before deletion
        videos      = self.list_user_videos(user_id)
        video_count = len(videos)
        frame_count = sum(v["frame_count"] for v in videos)

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))
                ])
            ),
        )

        log.info(
            "Admin: deleted user='%s' — %d videos, %d frames removed.",
            user_id, video_count, frame_count,
        )
        return {
            "deleted":        True,
            "user_id":        user_id,
            "videos_removed": video_count,
            "frames_removed": frame_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _count_for_filter(self, filt: qm.Filter) -> int:
        """Count points matching *filt* without fetching vector data."""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=filt,
                exact=True,
            )
            return result.count
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Singleton — lazy connection (won't crash the app if Qdrant is offline)
# ---------------------------------------------------------------------------
qdrant_service = QdrantService()
