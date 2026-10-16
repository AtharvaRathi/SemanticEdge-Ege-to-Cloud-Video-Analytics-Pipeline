from __future__ import annotations

"""
edge/context_firewall.py
========================
SemanticEdge Context Firewall — local edge-side novelty filter.

Architecture
------------
  KeyframePacket (from edge_ingest.py)
         │
         ▼
  ContextFirewall.evaluate()
         │
         ├── SigLIP image embedding (768-dim)
         │
         ├── L2-normalise → FAISS IndexFlatIP (cosine similarity via dot product)
         │
         ├── cosine_sim > threshold (default 0.95)?
         │       ├── YES → DROP (routine / seen before) — no bandwidth used
         │       └── NO  → FORWARD (novel event) ──────────────────────────────┐
         │                                                                      │
         └────────────────────────────────────────────────────────────────────┘
                        ▼
               HTTP POST to /api/v1/ingest   (cloud endpoint)
               — or — puts into a local forward_queue for custom consumer

FAISS usage
-----------
* `faiss.IndexFlatIP` — exact inner product on L2-normalised vectors = cosine sim
* IndexFlatIP does NOT store compressed vectors; it is exact by design.
* We maintain a baseline of N frames (configurable) and compare every new
  keyframe against those N reference vectors.  The maximum cosine similarity
  across the baseline set is used as the drop/forward decision.
* FAISS must be installed:  pip install faiss-cpu

SigLIP embedding on the edge
-----------------------------
Uses the HuggingFace `transformers` SigLIP vision encoder directly (no ONNX
required on the edge side).  The first call downloads weights (~350 MB);
subsequent calls use the HF cache.  Alternatively point --model-dir to a
local snapshot.

Usage (standalone — reads from a live EdgeIngestionPipeline queue)
-------------------------------------------------------------------
  python context_firewall.py --cloud-url http://localhost:8000

Usage (library — integrate with your own pipeline)
---------------------------------------------------
  from context_firewall import ContextFirewall, FirewallConfig

  cfg = FirewallConfig(similarity_threshold=0.95, baseline_size=5)
  fw  = ContextFirewall(cfg)
  fw.set_baseline_from_frames([frame1, frame2, frame3])

  decision = fw.evaluate_frame(packet)
  if decision.forwarded:
      ...  # send to cloud

"""

import argparse
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("context_firewall")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768          # google/siglip-base-patch16-224
DEFAULT_MODEL  = "google/siglip-base-patch16-224"
DEFAULT_THRESHOLD = 0.99     # cosine similarity above this → DROP


@dataclass
class FirewallConfig:
    """
    Configuration for the ContextFirewall.

    Parameters
    ----------
    similarity_threshold : Drop a keyframe if max cosine similarity with the
                           baseline index exceeds this value. [0, 1]
    baseline_size        : Number of frames to use when auto-building a baseline
                           from the first frames of a stream.
    model_name_or_path   : HuggingFace model ID or local snapshot directory.
    cloud_ingest_url     : Full URL of the cloud /ingest endpoint.
                           If empty, forwarded frames are placed in forward_queue.
    cloud_user_id        : user_id sent with each ingest request.
    cloud_source_id      : video_id / stream identifier sent as filename.
    forward_queue_maxsize: Maximum size of the local forward queue.
    request_timeout_s    : HTTP POST timeout when forwarding to the cloud.
    """
    similarity_threshold: float = DEFAULT_THRESHOLD
    baseline_size: int = 5
    model_name_or_path: str = DEFAULT_MODEL
    cloud_ingest_url: str = ""
    cloud_user_id: str = "edge_default"
    cloud_source_id: str = "edge_stream"
    forward_queue_maxsize: int = 128
    request_timeout_s: float = 5.0


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------

@dataclass
class FirewallDecision:
    """Result of a single firewall evaluation."""
    frame_idx: int
    timestamp: float
    cosine_similarity: float   # max similarity vs. baseline (–1 to 1)
    forwarded: bool            # True → sent to cloud; False → dropped
    reason: str                # human-readable decision rationale


# ---------------------------------------------------------------------------
# SigLIP Embedder
# ---------------------------------------------------------------------------

class SigLIPEdgeEmbedder:
    """
    Lightweight SigLIP image encoder for the edge.

    Wraps HuggingFace `transformers` so no ONNX export is needed.
    Outputs L2-normalised 768-dim float32 vectors, matching the cloud backend.
    """

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name_or_path
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    def _load(self) -> None:
        """Lazy-load the SigLIP model (thread-safe, only once)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            log.info("Loading SigLIP vision encoder: %s …", self._model_name)
            try:
                from transformers import AutoProcessor, AutoModel
                import torch

                self._processor = AutoProcessor.from_pretrained(self._model_name)
                self._model = AutoModel.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.float32,
                )
                self._model.eval()
                self._torch = torch
                log.info("SigLIP model loaded (vision encoder, 768-dim).")
            except Exception as exc:
                log.error("Failed to load SigLIP: %s", exc)
                raise

    # -----------------------------------------------------------------------
    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Compute a 768-dim L2-normalised embedding for a BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray  — OpenCV BGR frame (uint8, HxWx3)

        Returns
        -------
        np.ndarray of shape (768,) dtype float32, unit-norm.
        """
        self._load()

        # Convert BGR → RGB PIL Image
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self._processor(images=pil_image, return_tensors="pt")

        with self._torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # image_features: [1, 768]
        vec = image_features[0].cpu().numpy().astype(np.float32)

        # L2-normalise so dot product == cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec

    def embed_batch(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Embed a batch of BGR frames.

        Returns
        -------
        np.ndarray of shape (N, 768), each row unit-normalised.
        """
        self._load()

        import cv2
        from PIL import Image

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames_bgr
        ]

        inputs = self._processor(
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vecs = features.cpu().numpy().astype(np.float32)  # (N, 768)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return vecs / norms


# ---------------------------------------------------------------------------
# FAISS baseline index
# ---------------------------------------------------------------------------

class FAISSBaselineIndex:
    """
    Thin wrapper around faiss.IndexFlatIP for cosine-similarity baseline checks.

    Since vectors are L2-normalised before insertion, inner product == cosine
    similarity.  No approximate-nearest-neighbour needed; the baseline is small
    (≤ 50 vectors by default) so exact search is fast.

    Requires:  pip install faiss-cpu
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self._index = None
        self._n_vectors: int = 0

    # -----------------------------------------------------------------------
    def _build(self) -> None:
        """Lazy-import faiss and create the index."""
        if self._index is not None:
            return
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)
            log.info("FAISS IndexFlatIP created (dim=%d).", self.dim)
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed.  Run:  pip install faiss-cpu\n"
                "Alternatively set use_faiss=False to use pure-numpy fallback."
            )

    # -----------------------------------------------------------------------
    def add(self, vectors: np.ndarray) -> None:
        """
        Add L2-normalised vectors to the baseline index.

        Parameters
        ----------
        vectors : np.ndarray  shape (N, dim) or (dim,), float32
        """
        self._build()
        vectors = np.atleast_2d(vectors).astype(np.float32)
        assert vectors.shape[1] == self.dim, (
            f"Expected dim {self.dim}, got {vectors.shape[1]}"
        )
        self._index.add(vectors)
        self._n_vectors += len(vectors)
        log.debug("Baseline index now has %d vectors.", self._n_vectors)

    # -----------------------------------------------------------------------
    def max_cosine_similarity(self, query: np.ndarray) -> float:
        """
        Return the maximum cosine similarity between *query* and every
        vector in the baseline index.

        Returns –1.0 if the index is empty (no baseline set yet).
        """
        if self._n_vectors == 0 or self._index is None:
            return -1.0  # no baseline → treat as novel

        q = np.atleast_2d(query).astype(np.float32)   # (1, dim)
        # search returns shape (1, n_vectors); k capped at n_vectors
        k = min(self._n_vectors, 1)
        distances, _ = self._index.search(q, k)
        # distances[0] are inner products = cosine similarities (L2-normed)
        return float(np.max(distances[0]))

    @property
    def size(self) -> int:
        return self._n_vectors

    def reset(self) -> None:
        """Clear all baseline vectors."""
        if self._index is not None:
            self._index.reset()
        self._n_vectors = 0


# ---------------------------------------------------------------------------
# Pure-numpy fallback (no faiss needed)
# ---------------------------------------------------------------------------

class NumpyBaselineIndex:
    """
    Numpy-only cosine similarity baseline.  Slower than FAISS but zero deps.
    Automatically used if faiss-cpu is not installed.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None  # (N, dim)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.atleast_2d(vectors).astype(np.float32)
        self._vectors = (
            vectors if self._vectors is None
            else np.vstack([self._vectors, vectors])
        )

    def max_cosine_similarity(self, query: np.ndarray) -> float:
        if self._vectors is None or len(self._vectors) == 0:
            return -1.0
        q = query.astype(np.float32)
        # Cosine similarity = dot product for unit-normed vectors
        sims = self._vectors @ q          # (N,)
        return float(np.max(sims))

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def reset(self) -> None:
        self._vectors = None


# ---------------------------------------------------------------------------
# Context Firewall
# ---------------------------------------------------------------------------

class ContextFirewall:
    """
    Edge-side novelty filter.

    Workflow
    --------
    1. Set a baseline:
         fw.set_baseline_from_frames([frame1, frame2, ...])
       or load embeddings directly:
         fw.set_baseline_embeddings(embedding_array)

    2. For each keyframe from EdgeIngestionPipeline:
         decision = fw.evaluate_frame(packet)
         if decision.forwarded:
             ...  # send to cloud / write to disk

    3. (Optional) run the firewall in its own thread feeding from a queue:
         fw.start_consuming(input_queue)
    """

    def __init__(self, config: Optional[FirewallConfig] = None) -> None:
        self.cfg = config or FirewallConfig()
        self._embedder = SigLIPEdgeEmbedder(self.cfg.model_name_or_path)
        self._index = self._make_index()
        self._baseline_locked = False   # once locked, further set_baseline calls warn
        self._stop_event = threading.Event()

        # Output queue for forwarded packets
        self.forward_queue: queue.Queue = queue.Queue(
            maxsize=self.cfg.forward_queue_maxsize
        )

        # Stats
        self._stats = {
            "evaluated": 0,
            "forwarded": 0,
            "dropped_routine": 0,
            "dropped_queue_full": 0,
            "http_sent": 0,
            "http_errors": 0,
        }

        log.info(
            "ContextFirewall ready — threshold=%.2f baseline_size=%d",
            self.cfg.similarity_threshold,
            self.cfg.baseline_size,
        )

    # -----------------------------------------------------------------------
    # Baseline management
    # -----------------------------------------------------------------------

    def _make_index(self):
        """Return FAISS index if available, else numpy fallback."""
        try:
            import faiss  # noqa: F401
            return FAISSBaselineIndex(dim=EMBEDDING_DIM)
        except ImportError:
            log.warning(
                "faiss-cpu not found — using pure-numpy baseline (slower). "
                "Install with:  pip install faiss-cpu"
            )
            return NumpyBaselineIndex(dim=EMBEDDING_DIM)

    def set_baseline_from_frames(
        self, frames: List[np.ndarray], batch: bool = True
    ) -> None:
        """
        Compute embeddings for *frames* and load them as the baseline.
        Call once before starting the main consume loop.

        Parameters
        ----------
        frames : list of BGR numpy arrays
        batch  : if True (faster), embed all frames in one forward pass.
        """
        if not frames:
            raise ValueError("frames must be non-empty.")

        log.info("Building baseline from %d frames …", len(frames))
        if batch:
            embeddings = self._embedder.embed_batch(frames)
        else:
            embeddings = np.vstack([self._embedder.embed(f) for f in frames])

        self.set_baseline_embeddings(embeddings)

    def set_baseline_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Load pre-computed embeddings (shape (N, 768)) as the baseline.
        Vectors are assumed to be L2-normalised already.
        """
        embeddings = np.atleast_2d(embeddings).astype(np.float32)
        self._index.reset()
        self._index.add(embeddings)
        self._baseline_locked = True
        log.info("Baseline loaded — %d reference vectors.", self._index.size)

    def auto_baseline_from_queue(
        self,
        q: queue.Queue,
        n_frames: Optional[int] = None,
    ) -> None:
        """
        Consume the first *n_frames* packets from *q* to build a baseline,
        then hand back control.  Blocks until enough frames are collected.
        """
        n_frames = n_frames or self.cfg.baseline_size
        log.info("Collecting %d frames for auto-baseline …", n_frames)
        frames: List[np.ndarray] = []

        while len(frames) < n_frames:
            try:
                pkt = q.get(timeout=5.0)
                frames.append(pkt.frame)
            except queue.Empty:
                log.warning("Queue timeout while collecting baseline frames.")
                break

        if not frames:
            raise RuntimeError("Could not collect any frames for baseline.")

        self.set_baseline_from_frames(frames)

    # -----------------------------------------------------------------------
    # Single-frame evaluation
    # -----------------------------------------------------------------------

    def evaluate_frame(self, packet) -> FirewallDecision:
        """
        Evaluate one KeyframePacket through the firewall.

        Returns a FirewallDecision.  Forwarded packets are either sent to the
        cloud via HTTP or placed in self.forward_queue (if no URL configured).
        """
        self._stats["evaluated"] += 1

        # ── 1. Embed ─────────────────────────────────────────────────────────
        try:
            embedding = self._embedder.embed(packet.frame)
        except Exception as exc:
            log.error("Embedding failed for frame %d: %s", packet.frame_idx, exc)
            return FirewallDecision(
                frame_idx=packet.frame_idx,
                timestamp=packet.timestamp,
                cosine_similarity=-1.0,
                forwarded=False,
                reason=f"embedding_error: {exc}",
            )

        # ── 2. Compare against baseline ──────────────────────────────────────
        if self._index.size == 0:
            # No baseline yet — auto-add this frame to baseline and forward
            self._index.add(embedding)
            log.info(
                "No baseline set — frame %d added as initial baseline and forwarded.",
                packet.frame_idx,
            )
            cos_sim = -1.0  # nothing to compare against
            forwarded = True
            reason = "auto_baseline_seed"
        else:
            cos_sim = self._index.max_cosine_similarity(embedding)
            if cos_sim > self.cfg.similarity_threshold:
                forwarded = False
                reason = f"routine (cosine={cos_sim:.4f} > {self.cfg.similarity_threshold})"
                self._stats["dropped_routine"] += 1
            else:
                forwarded = True
                reason = f"novel (cosine={cos_sim:.4f} <= {self.cfg.similarity_threshold})"

        decision = FirewallDecision(
            frame_idx=packet.frame_idx,
            timestamp=packet.timestamp,
            cosine_similarity=cos_sim,
            forwarded=forwarded,
            reason=reason,
        )

        # ── 3. Forward if novel ──────────────────────────────────────────────
        if forwarded:
            self._stats["forwarded"] += 1
            self._dispatch(packet, embedding, decision)

        log.debug(
            "Frame %d — %s",
            packet.frame_idx,
            reason,
        )
        return decision

    # -----------------------------------------------------------------------
    # Dispatch (HTTP or queue)
    # -----------------------------------------------------------------------

    def _dispatch(self, packet, embedding: np.ndarray, decision: FirewallDecision) -> None:
        """Send a novel keyframe to the cloud or put it in the forward queue."""
        if self.cfg.cloud_ingest_url:
            self._http_forward(packet, embedding)
        else:
            try:
                self.forward_queue.put_nowait((packet, embedding))
            except queue.Full:
                self._stats["dropped_queue_full"] += 1
                log.warning(
                    "Forward queue full — frame %d dropped. "
                    "Check that your cloud consumer is keeping up.",
                    packet.frame_idx,
                )

    def _http_forward(self, packet, embedding: np.ndarray) -> None:
        """
        POST the keyframe and its embedding to the cloud /ingest endpoint.

        Payload
        -------
        JSON body (application/json):
          {
            "user_id":    "<str>",
            "video_id":   "<str>",
            "embedding":  [float, ...],   // 768 floats
            "timestamp":  <float>,
            "frame_idx":  <int>,
            "metadata":   { ... }
          }
        """
        import json
        import urllib.request
        import urllib.error
        import base64
        import cv2

        image_data = None
        if hasattr(packet, "frame") and packet.frame is not None and isinstance(packet.frame, np.ndarray) and packet.frame.size > 0:
            try:
                # Resize to max 640px width to keep payload reasonable (~15-35KB)
                h, w = packet.frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    frame_small = cv2.resize(packet.frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = packet.frame
                ret, buf = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    image_data = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            except Exception as e:
                log.warning("Failed to encode frame to JPEG: %s", e)

        payload = json.dumps({
            "user_id":   self.cfg.cloud_user_id,
            "video_id":  self.cfg.cloud_source_id,
            "embedding": embedding.tolist(),
            "timestamp": packet.timestamp,
            "frame_idx": packet.frame_idx,
            "image":     image_data,
            "metadata":  {
                **packet.metadata,
                "source":       packet.source,
                "object_count": packet.object_count,
                "motion_score": packet.motion_score,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            self.cfg.cloud_ingest_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self.cfg.request_timeout_s
            ) as resp:
                status = resp.status
                if status == 200:
                    self._stats["http_sent"] += 1
                    log.debug("Frame %d forwarded → HTTP %d", packet.frame_idx, status)
                else:
                    log.warning(
                        "Ingest endpoint returned HTTP %d for frame %d.",
                        status, packet.frame_idx,
                    )
        except urllib.error.URLError as exc:
            self._stats["http_errors"] += 1
            log.error("HTTP forward failed for frame %d: %s", packet.frame_idx, exc)

    # -----------------------------------------------------------------------
    # Threaded consumer loop
    # -----------------------------------------------------------------------

    def start_consuming(
        self,
        input_queue: queue.Queue,
        auto_baseline: bool = True,
    ) -> threading.Thread:
        """
        Start a daemon thread that continuously pulls KeyframePackets from
        *input_queue* and runs them through the firewall.

        Parameters
        ----------
        input_queue   : Queue fed by EdgeIngestionPipeline (KeyframePacket items).
        auto_baseline : If True, the first `cfg.baseline_size` packets are used
                        to build the baseline before filtering begins.

        Returns the running Thread.
        """
        def _loop() -> None:
            if auto_baseline and self._index.size == 0:
                try:
                    self.auto_baseline_from_queue(input_queue)
                except RuntimeError as exc:
                    log.error("Auto-baseline failed: %s — continuing without baseline.", exc)

            log.info("Context firewall consume loop started.")
            while not self._stop_event.is_set():
                try:
                    packet = input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                self.evaluate_frame(packet)

                if self._stats["evaluated"] % 100 == 0:
                    log.info("Firewall stats: %s", self._stats)

        t = threading.Thread(target=_loop, daemon=True, name="context-firewall")
        t.start()
        log.info("Firewall consumer thread started.")
        return t

    def stop(self) -> None:
        """Signal the consume loop to exit."""
        self._stop_event.set()

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SemanticEdge Context Firewall",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", required=True,
        help="RTSP URL or MP4 file path (passed to EdgeIngestionPipeline).",
    )
    p.add_argument(
        "--cloud-url", default="",
        help=(
            "Full URL of the cloud /ingest endpoint "
            "(e.g. http://localhost:8000/api/v1/ingest). "
            "Leave empty to only print decisions."
        ),
    )
    p.add_argument(
        "--user-id", default="edge_default",
        help="user_id tag sent with each forwarded frame.",
    )
    p.add_argument(
        "--source-id", default="edge_stream",
        help="video_id / stream identifier sent with each forwarded frame.",
    )
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Cosine similarity threshold above which frames are dropped.",
    )
    p.add_argument(
        "--baseline-size", type=int, default=5,
        help="Number of initial frames used to build the baseline.",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace SigLIP model name or local snapshot path.",
    )
    p.add_argument(
        "--weights", default="yolov8n.pt",
        help="YOLOv8 weights for the upstream edge ingest pipeline.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Import edge ingest pipeline ─────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from edge_ingest import EdgeIngestionPipeline

    shared_queue: queue.Queue = queue.Queue(maxsize=64)

    ingest = EdgeIngestionPipeline(
        source=args.source,
        weights_path=args.weights,
        output_queue=shared_queue,
    )

    cfg = FirewallConfig(
        similarity_threshold=args.threshold,
        baseline_size=args.baseline_size,
        model_name_or_path=args.model,
        cloud_ingest_url=args.cloud_url,
        cloud_user_id=args.user_id,
        cloud_source_id=args.source_id,
    )
    firewall = ContextFirewall(cfg)

    # Start ingest in background thread
    ingest_thread = ingest.run_in_thread()

    # Start firewall consumer (will auto-build baseline from first N frames)
    fw_thread = firewall.start_consuming(shared_queue, auto_baseline=True)

    try:
        ingest_thread.join()
    except KeyboardInterrupt:
        log.info("Interrupted — stopping.")
        ingest.stop()
        firewall.stop()

    log.info("Final firewall stats: %s", firewall.stats)


if __name__ == "__main__":
    main()
