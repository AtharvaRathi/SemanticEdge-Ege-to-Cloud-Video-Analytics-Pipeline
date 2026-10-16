"""
app/api/routes.py
=================
FastAPI route handlers.

All model inference goes through SigLipEngine (ONNX Runtime) – no PyTorch
is needed at request time, keeping the dependency footprint small on CPU-only
deployments.
"""

import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.engine.models import SigLipEngine
from app.services.qdrant import qdrant_service
from app.services.tasks import process_video_background, tasks_db

log = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_EMBEDDING_DIM = 768  # google/siglip-base-patch16-224


# ---------------------------------------------------------------------------
# Pydantic models for the /ingest endpoint
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """
    Payload sent by the edge Context Firewall for a single novel keyframe.

    Fields
    ------
    user_id   : Tenant / stream owner identifier (used for multi-tenant filtering).
    video_id  : Logical stream or camera identifier (groups frames under one source).
    embedding : Pre-computed SigLIP image embedding from the edge — must be exactly
                768 floats, L2-normalised (the firewall ensures this).
    timestamp : UNIX epoch float (seconds) at the moment the frame was captured.
    frame_idx : Sequential frame index within the source stream.
    metadata  : Arbitrary dict forwarded verbatim into the Qdrant payload
                (e.g. object counts, tracker IDs, motion score, source URL).
    """
    user_id:   str = Field(..., min_length=1, max_length=128)
    video_id:  str = Field(..., min_length=1, max_length=256)
    embedding: List[float] = Field(..., description="768-dim SigLIP embedding (unit-normed)")
    timestamp: float = Field(..., gt=0, description="UNIX epoch seconds")
    frame_idx: int   = Field(..., ge=0)
    image:     Optional[str] = Field(default=None, description="Base64 encoded JPEG or static URL")
    metadata:  Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("embedding")
    @classmethod
    def _check_dim(cls, v: List[float]) -> List[float]:
        if len(v) != _EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have exactly {_EMBEDDING_DIM} floats, got {len(v)}."
            )
        return v


class IngestResponse(BaseModel):
    status:    str
    frame_idx: int
    video_id:  str
    user_id:   str
    timestamp: float


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """
    Returns Qdrant connectivity status, mode, and model availability.
    Hit this first when debugging connection issues.
    """
    qdrant_info = qdrant_service.health_check()

    split_vision = os.path.join(settings.ONNX_MODEL_DIR, settings.ONNX_VISION_MODEL_FILE)
    split_text   = os.path.join(settings.ONNX_MODEL_DIR, settings.ONNX_TEXT_MODEL_FILE)
    combined     = os.path.join(settings.ONNX_MODEL_DIR, settings.ONNX_MODEL_FILE)

    if os.path.exists(split_vision) and os.path.exists(split_text):
        model_status = "ok (split)"
        model_path = f"{settings.ONNX_VISION_MODEL_FILE}, {settings.ONNX_TEXT_MODEL_FILE}"
    elif os.path.exists(combined):
        model_status = "ok (combined)"
        model_path = settings.ONNX_MODEL_FILE
    else:
        model_status = "missing"
        model_path = combined

    return {
        "api": "ok",
        "qdrant": qdrant_info,
        "model": {
            "status": model_status,
            "path": model_path,
        },
    }


# ---------------------------------------------------------------------------
# Edge ingest — receives pre-computed keyframe embeddings from the firewall
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse)
async def ingest_keyframe(body: IngestRequest):
    """
    Receive a single novel keyframe embedding from the edge Context Firewall
    and store it in the Qdrant vector index.

    The edge firewall:
      1. Runs YOLOv8 + ByteTrack + Adaptive Frame Sampling.
      2. Embeds each keyframe with SigLIP (768-dim, L2-normalised).
      3. Compares it against a local FAISS baseline; drops routine frames.
      4. POSTs only the *novel* frames here.

    This endpoint skips server-side embedding — the 768-dim vector is already
    computed and normalised by the edge.  This keeps cloud CPU cost near zero
    for ingestion.

    Request body (JSON)
    -------------------
    {
        "user_id":   "camera_01",
        "video_id":  "rtsp_stream_2026-10-17",
        "embedding": [0.021, -0.003, ...],   // 768 floats, L2-normalised
        "timestamp": 1729100000.123,
        "frame_idx": 472,
        "metadata":  { "object_count": 3, "motion_score": 0.42, ... }
    }

    Returns
    -------
    200 OK with IngestResponse on success.
    422 if embedding dimension != 768.
    503 if Qdrant is unavailable.
    """
    log.info(
        "Ingest request — user=%s video=%s frame=%d ts=%.3f",
        body.user_id, body.video_id, body.frame_idx, body.timestamp,
    )

    # ── Validate and prep the embedding ─────────────────────────────────────
    embedding_np = np.array(body.embedding, dtype=np.float32)  # (768,)

    # Defensive re-normalisation: the edge should send unit vectors but
    # floating-point serialisation can introduce tiny drift.
    norm = float(np.linalg.norm(embedding_np))
    if norm < 1e-8:
        raise HTTPException(
            status_code=422,
            detail="Received a zero-norm embedding vector — cannot store.",
        )
    embedding_np = embedding_np / norm

    # ── Build the payload metadata ──────────────────────────────────────────
    # Merge caller-supplied metadata with required top-level fields so that
    # Qdrant payload has both indexed fields (user_id, video_id) and the
    # full metadata blob for result enrichment.
    extra_metadata = {k: v for k, v in (body.metadata or {}).items()}
    if body.image:
        extra_metadata["image"] = body.image

    # ── Upsert into Qdrant ──────────────────────────────────────────────────
    # upsert_vectors() expects list-of-lists; wrap accordingly.
    try:
        qdrant_service.upsert_vectors(
            user_id=body.user_id,
            video_id=body.video_id,
            vectors=[embedding_np.tolist()],
            timestamps=[body.timestamp],
            filename=body.video_id,      # use video_id as the logical filename
            frame_indices=[body.frame_idx],
            metadata_list=[extra_metadata] if extra_metadata else None,
        )
    except Exception as exc:
        log.error(
            "Qdrant upsert failed for frame %d (user=%s, video=%s): %s",
            body.frame_idx, body.user_id, body.video_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Failed to store embedding in Qdrant: {exc}",
        )

    log.info(
        "Keyframe stored — user=%s video=%s frame=%d",
        body.user_id, body.video_id, body.frame_idx,
    )
    return IngestResponse(
        status="stored",
        frame_idx=body.frame_idx,
        video_id=body.video_id,
        user_id=body.user_id,
        timestamp=body.timestamp,
    )


# ---------------------------------------------------------------------------
# Debug — Phase 3 observability
# ---------------------------------------------------------------------------

@router.get("/debug/collection")
async def debug_collection():
    """
    Quick diagnostic: shows how many vectors are stored in Qdrant.
    Use this after uploading a video to confirm embeddings were actually stored.
    Also runs a sample ONNX inference to confirm text-embedding dimension.
    """
    # ── Qdrant point count ────────────────────────────────────────────────────
    try:
        qdrant_service._ensure_collection()
        info = qdrant_service.client.get_collection(settings.QDRANT_COLLECTION)
        qdrant_info = {
            "collection": settings.QDRANT_COLLECTION,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as exc:
        qdrant_info = {"error": str(exc)}

    # ── ONNX model info ───────────────────────────────────────────────────────
    model_info = {"status": "not_checked"}
    try:
        siglip = SigLipEngine()
        import numpy as _np
        dummy_ids  = _np.zeros((1, 64), dtype=_np.int64)
        dummy_mask = _np.ones((1, 64),  dtype=_np.int64)
        emb = siglip.get_text_features(dummy_ids, dummy_mask)
        model_info = {
            "status": "ok",
            "mode": "split" if siglip.is_split_mode else "combined",
            "text_embedding_dim": int(emb.shape[-1]),
            "onnx_inputs":  sorted(siglip._input_names),
            "onnx_outputs": siglip._output_names,
        }
        if siglip.is_split_mode:
            model_info["vision_inputs"]  = sorted(siglip._vision_inputs)
            model_info["vision_outputs"] = siglip._vision_outputs
            model_info["text_inputs"]    = sorted(siglip._text_inputs)
            model_info["text_outputs"]   = siglip._text_outputs
    except Exception as exc:
        model_info = {"status": "error", "detail": str(exc)}

    return {"qdrant": qdrant_info, "model": model_info}


# ---------------------------------------------------------------------------
# Upload & ingestion
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_videos(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Accept one OR more video files and start a background processing task
    for each.  All tasks run concurrently — useful for batch ingestion.

    Returns a list of task objects so the client can poll each independently.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    for file in files:
        # Sanitise extension; default to mp4 if filename has no dot
        raw_name = file.filename or "upload"
        ext = raw_name.rsplit(".", 1)[-1] if "." in raw_name else "mp4"
        task_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_DIR, f"{task_id}.{ext}")

        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        tasks_db[task_id] = {"status": "queued", "filename": raw_name}
        background_tasks.add_task(process_video_background, task_id, temp_path, user_id)

        results.append({"task_id": task_id, "filename": raw_name, "status": "queued"})
        log.info("Queued task %s for file '%s' (user=%s)", task_id, raw_name, user_id)

    return {"tasks": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    status = tasks_db.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


def _generate_placeholder_svg(video_id: str, frame_idx: int, score: float) -> str:
    """Generate a sleek dark-themed SVG preview if an explicit image frame is not stored."""
    import urllib.parse
    pct = int(max(0, min(100, score * 100)))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360" fill="none">
<rect width="640" height="360" fill="#0d1117"/>
<rect x="20" y="20" width="600" height="320" rx="10" fill="#161b22" stroke="#30363d" stroke-width="2"/>
<circle cx="320" cy="140" r="38" fill="#21262d" stroke="#388bfd" stroke-width="2"/>
<path d="M306 130h28a4 4 0 0 1 4 4v16a4 4 0 0 1-4 4h-28a4 4 0 0 1-4-4v-16a4 4 0 0 1 4-4zm32 6l14-8v28l-14-8v-12z" fill="#58a6ff"/>
<text x="320" y="215" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif" font-size="15" font-weight="600" fill="#f0f6fc" text-anchor="middle">{video_id}</text>
<text x="320" y="242" font-family="monospace" font-size="13" fill="#8b949e" text-anchor="middle">Frame #{frame_idx} • Match: {pct}%</text>
</svg>"""
    return f"data:image/svg+xml;utf8,{urllib.parse.quote(svg)}"


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

@router.post("/search")
async def search_video(
    request: Request,
    query: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    limit: int = 5,
    min_score: float = 0.0,
):
    """
    Embed *query* with the SigLIP text encoder (ONNX) and search Qdrant.

    Accepts parameters either via Form-encoded data or JSON body.
    Ensures SigLIP vectors are L2-normalized and filters results by min_score.
    Returns a list of results sorted by relevance score (highest first).
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                query = body.get("query", query)
                user_id = body.get("user_id", user_id)
                limit = int(body.get("limit", limit))
                if "min_score" in body:
                    min_score = float(body.get("min_score"))
        except Exception:
            pass

    if not query:
        raise HTTPException(status_code=422, detail="Missing required field: 'query'")
    if not user_id:
        user_id = "edge_default"

    log.info("Search request: query='%s', user_id='%s', limit=%d, min_score=%.2f", query, user_id, limit, min_score)

    # ── 1. Load model (singleton – already in memory after first call) ──────
    try:
        siglip = SigLipEngine()
        _, processor, _ = siglip.get_components()
    except Exception as exc:
        log.error("Failed to load SigLipEngine: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Did you run tools/quantize_model.py? Error: {exc}",
        )

    # ── 2. Tokenise query ────────────────────────────────────────────────────
    try:
        inputs = processor(
            text=[query],
            return_tensors="np",         
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        input_ids = inputs["input_ids"].astype(np.int64)

        # attention_mask is required by SigLIP tokenizer
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.astype(np.int64)

        log.debug(
            "Tokenised query: input_ids shape=%s, attention_mask shape=%s",
            input_ids.shape,
            attention_mask.shape if attention_mask is not None else "None",
        )

    except Exception as exc:
        log.error("Tokenisation failed for query '%s': %s", query, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tokenisation failed: {exc}")

    # ── 3. Run ONNX text encoder & L2-normalize ─────────────────────────────
    try:
        text_embedding = siglip.get_text_features(input_ids, attention_mask)
        # text_embedding shape: [1, 768]  →  take first row and guarantee L2 unit norm
        text_vec_np = np.asarray(text_embedding[0], dtype=np.float32)
        norm = float(np.linalg.norm(text_vec_np))
        if norm > 1e-8:
            text_vec_np = text_vec_np / norm
        text_vector = text_vec_np.tolist()  # Python list of 768 unit floats

        log.debug(
            "Text embedding generated (L2-normed): dim=%d, norm=%.4f, first5=%s",
            len(text_vector),
            norm,
            [f"{v:.4f}" for v in text_vector[:5]],
        )

    except Exception as exc:
        log.error("ONNX inference failed for query '%s': %s", query, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    # ── 4. Vector search in Qdrant ───────────────────────────────────────────
    try:
        results = qdrant_service.search(user_id, text_vector, limit)
    except Exception as exc:
        log.error("Qdrant search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Vector search failed: {exc}")

    # ── 5. Filter by min_score threshold and format ─────────────────────────
    filtered_results = [
        res for res in results
        if res.score is not None and float(res.score) >= min_score
    ]

    if not filtered_results:
        log.info(
            "No results passed min_score=%.2f threshold for query='%s', user_id='%s'. (Raw hits: %d)",
            min_score, query, user_id, len(results),
        )

    return [
        {
            "score":     round(float(res.score), 4),
            "timestamp": res.payload.get("timestamp"),
            "frame_idx": res.payload.get("frame_idx"),
            "video_id":  res.payload.get("video_id"),
            "filename":  res.payload.get("filename", ""),
            "image":     res.payload.get("image") or _generate_placeholder_svg(
                res.payload.get("video_id", "stream"),
                res.payload.get("frame_idx", 0),
                float(res.score)
            ),
            "metadata":  {
                k: v for k, v in res.payload.items()
                if k not in ("user_id", "video_id", "timestamp", "frame_idx", "filename", "image")
            },
        }
        for res in filtered_results
    ]


# ---------------------------------------------------------------------------
# Video management
# ---------------------------------------------------------------------------

@router.get("/videos")
async def list_videos(user_id: str = Query(..., description="User whose video library to list")):
    """
    List all videos indexed for *user_id*, with frame counts and video IDs.
    Use the returned video_id values with DELETE /video/{video_id} to remove videos.
    """
    try:
        videos = qdrant_service.list_user_videos(user_id)
    except Exception as exc:
        log.error("list_user_videos failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not list videos: {exc}")

    return {"user_id": user_id, "videos": videos, "count": len(videos)}


@router.delete("/video/{video_id}")
async def delete_video(
    video_id: str,
    user_id: str = Query(..., description="Owner of the video — only their data is touched"),
):
    """
    Delete all embedding vectors for *video_id* that belong to *user_id*.

    Safe to call on a video_id that no longer exists (returns frames_removed=0).
    Does NOT delete the original video file — only the search index entries.
    """
    try:
        result = qdrant_service.delete_video(user_id=user_id, video_id=video_id)
    except Exception as exc:
        log.error("delete_video failed for video_id='%s': %s", video_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Delete failed: {exc}")

    log.info("Deleted video_id='%s' for user='%s'.", video_id, user_id)
    return result


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------
# NOTE: These endpoints have no authentication in the prototype.
#       Before moving to production, add an API key or JWT middleware.
# ---------------------------------------------------------------------------

@router.get("/admin/users")
async def admin_list_users():
    """
    Admin: list every user_id in the collection with their video count and
    total indexed frames.  Requires a full collection scan — fast for prototype
    scale, consider a metadata sidecar for very large deployments.
    """
    try:
        users = qdrant_service.list_users()
    except Exception as exc:
        log.error("admin_list_users failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not list users: {exc}")

    return {"users": users, "total_users": len(users)}


@router.delete("/admin/user/{user_id}")
async def admin_delete_user(user_id: str):
    """
    Admin: permanently delete ALL data for *user_id* — every video, every
    embedding frame.  This operation is IRREVERSIBLE.

    Returns a summary: how many videos and frames were removed.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    try:
        result = qdrant_service.delete_user(user_id=user_id)
    except Exception as exc:
        log.error("admin_delete_user failed for user='%s': %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Delete failed: {exc}")

    return result
