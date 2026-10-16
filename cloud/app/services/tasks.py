from __future__ import annotations

import logging
import os
import uuid
from queue import Queue

from app.engine.pipeline import VideoFrameProducer, InferenceConsumer
from app.services.qdrant import qdrant_service

log = logging.getLogger(__name__)

# Simple in-memory task-status store.
# NOTE: Restarting the server clears this — acceptable for prototype scope.
tasks_db: dict = {}


def update_task_status(task_id: str, status: str, error: str | None = None) -> None:
    # Merge into existing entry so fields set at queue time (e.g. filename) are preserved
    entry = tasks_db.get(task_id, {})
    entry["status"] = status
    entry["error"]  = error
    tasks_db[task_id] = entry
    log.info("Task %s → %s", task_id, status)


def process_video_background(task_id: str, video_path: str, user_id: str) -> None:
    """
    Full ingestion pipeline run as a FastAPI background task:
      1. VideoFrameProducer  — FFmpeg single-pass scene extraction
      2. InferenceConsumer   — SigLIP INT8 batch embedding
      3. Qdrant upsert       — store vectors with metadata

    Errors are caught and surfaced in tasks_db so the /status endpoint
    can report them to the client.
    """
    filename = os.path.basename(video_path)

    try:
        update_task_status(task_id, "processing")

        # Bounded queue — limits RAM usage when producer outpaces consumer.
        # If consumer crashes, the producer will block here and NOT hang
        # forever because VideoFrameProducer is a daemon thread (it will be
        # killed when the process exits). For an extra safety net, see the
        # timeout on frame_queue.get() in InferenceConsumer.
        frame_queue: Queue = Queue(maxsize=64)

        producer = VideoFrameProducer(video_path, frame_queue)
        consumer = InferenceConsumer(frame_queue)

        producer.start()
        embeddings = consumer.process_video()   # blocks until sentinel received

        if not embeddings:
            # Always clean up temp file before returning
            _cleanup(video_path)
            update_task_status(task_id, "failed", "No embeddings generated. "
                               "Check that FFmpeg is on PATH and the video is valid.")
            return

        video_id = str(uuid.uuid4())
        vectors       = [e["vector"]    for e in embeddings]
        timestamps    = [e["timestamp"] for e in embeddings]
        frame_indices = [e["frame_idx"] for e in embeddings]

        qdrant_service.upsert_vectors(
            user_id=user_id,
            video_id=video_id,
            vectors=vectors,
            timestamps=timestamps,
            filename=filename,
            frame_indices=frame_indices,
        )

        _cleanup(video_path)
        update_task_status(task_id, "completed")
        log.info(
            "Task %s completed | video_id=%s | frames=%d | file='%s'",
            task_id, video_id, len(embeddings), filename,
        )

    except Exception as exc:
        log.error("Task %s failed: %s", task_id, exc, exc_info=True)
        _cleanup(video_path)
        update_task_status(task_id, "failed", str(exc))


def _cleanup(video_path: str) -> None:
    """Remove the temporary upload file after processing (success or failure)."""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            log.debug("Cleaned up temp file: %s", video_path)
    except OSError as exc:
        log.warning("Could not remove temp file '%s': %s", video_path, exc)
