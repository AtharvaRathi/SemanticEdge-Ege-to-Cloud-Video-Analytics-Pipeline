from __future__ import annotations

"""
edge/edge_ingest.py
===================
Edge-side ingestion pipeline for SemanticEdge.

Ingests an RTSP stream or MP4 file using OpenCV, runs YOLOv8 + ByteTrack
object detection/tracking, and applies Adaptive Frame Sampling to extract
only semantically meaningful keyframes.

Adaptive Frame Sampling logic
------------------------------
Two signals drive the sampling decision:

  1. Motion intensity  — inter-frame pixel difference (absolute diff mean).
     High motion implies scene activity worth capturing.

  2. Object count delta — change in the number of tracked objects between
     consecutive frames.  A sudden appearance or disappearance of objects
     is semantically significant even without pixel-level motion.

A dynamic cooldown prevents back-to-back keyframes on sustained motion;
the minimum gap shrinks when activity is high and grows when the scene
is static, adaptively throttling bandwidth.

Output
------
Keyframes are placed in a thread-safe queue as `KeyframePacket` namedtuples
so a downstream consumer (e.g. context_firewall.py) can pull them without
blocking the capture loop.

Usage (CLI)
-----------
  python edge_ingest.py --source rtsp://192.168.1.5/stream1
  python edge_ingest.py --source video.mp4 --weights yolov8n.pt --show

"""

import argparse
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("edge_ingest")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class KeyframePacket:
    """
    A single keyframe forwarded to the context firewall.

    Attributes
    ----------
    frame       : BGR numpy array (HxWx3, uint8).
    timestamp   : Wall-clock time of capture (float, seconds since epoch).
    frame_idx   : Sequential frame index within the source.
    source      : Source identifier (path or RTSP URL).
    object_count: Number of tracked objects at the moment of capture.
    motion_score: Normalised motion intensity [0, 1].
    metadata    : Arbitrary extra fields (detection labels, track IDs, etc.).
    """
    frame: np.ndarray
    timestamp: float
    frame_idx: int
    source: str
    object_count: int
    motion_score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adaptive frame sampler
# ---------------------------------------------------------------------------

class AdaptiveFrameSampler:
    """
    Decides whether each incoming frame should be kept as a keyframe.

    Parameters
    ----------
    motion_threshold   : Pixel-diff mean above which a frame is considered
                         to have significant motion (0–255 scale).
    object_delta_thresh: Minimum change in tracked-object count that forces
                         a keyframe regardless of motion score.
    min_cooldown_frames: Minimum number of frames to skip after a keyframe
                         (prevents burst sampling on sustained motion).
    max_cooldown_frames: Maximum cooldown applied when the scene is static.
    """

    def __init__(
        self,
        motion_threshold: float = 8.0,
        object_delta_thresh: int = 1,
        min_cooldown_frames: int = 3,
        max_cooldown_frames: int = 30,
    ) -> None:
        self.motion_threshold = motion_threshold
        self.object_delta_thresh = object_delta_thresh
        self.min_cooldown = min_cooldown_frames
        self.max_cooldown = max_cooldown_frames

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_object_count: int = 0
        self._cooldown_counter: int = 0
        self._current_cooldown: int = min_cooldown_frames

    def evaluate(
        self,
        frame: np.ndarray,
        object_count: int,
    ) -> tuple[bool, float]:
        """
        Evaluate whether *frame* is a keyframe.

        Returns
        -------
        is_keyframe : bool
        motion_score: float in [0, 1] — raw motion intensity normalised to 255.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0

        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            motion_score = float(diff.mean()) / 255.0  # normalise to [0,1]

        self._prev_gray = gray

        object_delta = abs(object_count - self._prev_object_count)
        self._prev_object_count = object_count

        # ── Determine if this frame qualifies ───────────────────────────────
        motion_triggered = motion_score * 255.0 >= self.motion_threshold
        object_triggered = object_delta >= self.object_delta_thresh

        is_candidate = motion_triggered or object_triggered

        # ── Apply cooldown ──────────────────────────────────────────────────
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            if not is_candidate:
                return False, motion_score

        if not is_candidate:
            # Gradually increase cooldown on static scenes
            self._current_cooldown = min(
                self._current_cooldown + 1, self.max_cooldown
            )
            return False, motion_score

        # ── Keyframe accepted — recalculate adaptive cooldown ──────────────
        # High motion → short cooldown (capture fast-changing scene more often)
        # Low motion  → longer cooldown (sparse changes need fewer frames)
        normalised_motion = min(motion_score * 255.0 / self.motion_threshold, 1.0)
        self._current_cooldown = max(
            self.min_cooldown,
            int(self.max_cooldown * (1.0 - normalised_motion)),
        )
        self._cooldown_counter = self._current_cooldown
        return True, motion_score


# ---------------------------------------------------------------------------
# Core ingestion pipeline
# ---------------------------------------------------------------------------

class EdgeIngestionPipeline:
    """
    Continuously reads frames from an RTSP stream or video file, runs
    YOLOv8 + ByteTrack, and emits keyframes via an output queue.

    Parameters
    ----------
    source          : RTSP URL or local video path.
    weights_path    : YOLOv8 weights file (.pt or .onnx).
    confidence      : Detection confidence threshold.
    iou_threshold   : NMS IoU threshold for YOLOv8.
    device          : Inference device ("cpu", "cuda:0", etc.).
    output_queue    : Thread-safe queue for KeyframePackets.
                      If None a default maxsize=64 queue is created.
    show            : Display live annotated video in a window.
    sampler_kwargs  : Keyword args forwarded to AdaptiveFrameSampler.
    """

    def __init__(
        self,
        source: str,
        weights_path: str = "yolov8n.pt",
        confidence: float = 0.3,
        iou_threshold: float = 0.7,
        device: str = "cpu",
        output_queue: Optional[queue.Queue] = None,
        show: bool = False,
        sampler_kwargs: Optional[dict] = None,
    ) -> None:
        self.source = source
        self.show = show
        self.output_queue: queue.Queue[KeyframePacket] = (
            output_queue if output_queue is not None else queue.Queue(maxsize=64)
        )

        # ── Tracking pipeline ────────────────────────────────────────────────
        # Import here to keep startup fast when module is imported as a library
        sys.path.insert(0, str(Path(__file__).parent))
        from tracking.yolov8_bytetrack import YOLOv8ByteTrackPipeline

        self._tracker = YOLOv8ByteTrackPipeline(
            weights_path=weights_path,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            device=device,
        )

        # ── Adaptive sampler ─────────────────────────────────────────────────
        self._sampler = AdaptiveFrameSampler(**(sampler_kwargs or {}))

        # ── State ────────────────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._frame_idx: int = 0
        self._stats = {
            "frames_read": 0,
            "keyframes_emitted": 0,
            "frames_dropped_sampler": 0,
            "queue_full_drops": 0,
        }

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @property
    def output_queue_ref(self) -> queue.Queue:
        """Return the output queue for downstream consumers."""
        return self.output_queue

    def run(self) -> None:
        """
        Block and run the ingestion loop until the stream ends or stop() is
        called.  Call from a dedicated thread to keep the main thread free.
        """
        cap = self._open_capture()
        if cap is None:
            return

        log.info("Ingestion started — source: %s", self.source)

        try:
            self._loop(cap)
        finally:
            cap.release()
            if self.show:
                cv2.destroyAllWindows()
            log.info(
                "Ingestion finished — %s",
                {k: v for k, v in self._stats.items()},
            )

    def run_in_thread(self) -> threading.Thread:
        """
        Spawn a daemon thread running ``run()`` and return it.
        The caller owns the thread; call ``stop()`` to request termination.
        """
        t = threading.Thread(target=self.run, daemon=True, name="edge-ingest")
        t.start()
        return t

    def stop(self) -> None:
        """Signal the ingestion loop to exit cleanly."""
        self._stop_event.set()

    @property
    def stats(self) -> dict:
        """Live statistics snapshot (read-only copy)."""
        return dict(self._stats)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open a VideoCapture and validate it."""
        source: Union[str, int] = self.source
        # Treat a bare integer string as a device index (webcam)
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            log.error(
                "Cannot open source '%s'. Check path, URL, or device index.",
                self.source,
            )
            return None

        # ── RTSP-specific tuning ────────────────────────────────────────────
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimal buffer → low latency
            log.info("RTSP source detected — CAP_PROP_BUFFERSIZE set to 1.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Stream: %dx%d @ %.1f fps", w, h, fps or 0)
        return cap

    def _loop(self, cap: cv2.VideoCapture) -> None:
        """Main capture-and-process loop."""
        # Determine if source is a live stream (RTSP or webcam device index) vs local video file
        is_live = (
            isinstance(self.source, int)
            or (isinstance(self.source, str) and (self.source.isdigit() or self.source.lower().startswith("rtsp")))
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = (1.0 / fps) if (fps and fps > 0 and not np.isnan(fps)) else (1.0 / 25.0)

        if not is_live:
            log.info(
                "Local video file detected. Enforcing real-time pacing (%.1f fps -> %.3fs delay per frame).",
                fps if (fps and fps > 0) else 25.0,
                frame_delay,
            )

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # End of file or dropped RTSP frame
                if is_live:
                    log.warning("Frame read failed — retrying in 100 ms…")
                    time.sleep(0.1)
                    continue
                log.info("End of video file reached.")
                break

            self._stats["frames_read"] += 1
            timestamp = time.time()

            # ── Detection + tracking ─────────────────────────────────────────
            try:
                detections, annotated = self._tracker.process_frame(
                    frame, annotate=self.show
                )
            except Exception as exc:
                log.error("Tracking error on frame %d: %s", self._frame_idx, exc)
                self._frame_idx += 1
                continue

            object_count = (
                len(detections.tracker_id)
                if detections.tracker_id is not None
                else len(detections)
            )

            # ── Adaptive sampling ────────────────────────────────────────────
            is_keyframe, motion_score = self._sampler.evaluate(frame, object_count)

            if is_keyframe:
                packet = KeyframePacket(
                    frame=frame.copy(),
                    timestamp=timestamp,
                    frame_idx=self._frame_idx,
                    source=self.source,
                    object_count=object_count,
                    motion_score=motion_score,
                    metadata={
                        "tracker_ids": (
                            detections.tracker_id.tolist()
                            if detections.tracker_id is not None
                            else []
                        ),
                        "class_ids": (
                            detections.class_id.tolist()
                            if detections.class_id is not None
                            else []
                        ),
                        "confidences": (
                            detections.confidence.tolist()
                            if detections.confidence is not None
                            else []
                        ),
                    },
                )
                try:
                    self.output_queue.put_nowait(packet)
                    self._stats["keyframes_emitted"] += 1
                    log.debug(
                        "Keyframe #%d — motion=%.3f objects=%d",
                        self._frame_idx, motion_score, object_count,
                    )
                except queue.Full:
                    self._stats["queue_full_drops"] += 1
                    log.warning(
                        "Output queue full — keyframe %d dropped. "
                        "Consider increasing queue size or speeding up consumer.",
                        self._frame_idx,
                    )
            else:
                self._stats["frames_dropped_sampler"] += 1

            # ── Optional display ─────────────────────────────────────────────
            if self.show and annotated is not None:
                cv2.imshow("SemanticEdge — Edge Ingest", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Display window closed by user.")
                    self.stop()
                    break

            # ── Real-time pacing for local video files ───────────────────────
            if not is_live:
                time.sleep(frame_delay)

            self._frame_idx += 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SemanticEdge edge ingestion pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", required=True,
        help="RTSP URL (rtsp://...) or path to an MP4 / video file.",
    )
    p.add_argument(
        "--weights", default="yolov8n.pt",
        help="YOLOv8 model weights file.",
    )
    p.add_argument(
        "--confidence", type=float, default=0.3,
        help="YOLOv8 detection confidence threshold.",
    )
    p.add_argument(
        "--iou", type=float, default=0.7,
        help="NMS IoU threshold.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Inference device: 'cpu', 'cuda:0', etc.",
    )
    p.add_argument(
        "--motion-threshold", type=float, default=8.0,
        help="Pixel-diff mean threshold for motion detection.",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Show annotated video in a live window.",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    pipeline = EdgeIngestionPipeline(
        source=args.source,
        weights_path=args.weights,
        confidence=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
        show=args.show,
        sampler_kwargs={"motion_threshold": args.motion_threshold},
    )

    # ── Simple consumer: print keyframe metadata to stdout ──────────────────
    def _print_consumer(q: queue.Queue) -> None:
        while True:
            try:
                pkt: KeyframePacket = q.get(timeout=2.0)
                log.info(
                    "Keyframe received — idx=%d ts=%.3f motion=%.3f objects=%d",
                    pkt.frame_idx, pkt.timestamp, pkt.motion_score, pkt.object_count,
                )
            except queue.Empty:
                if pipeline._stop_event.is_set():
                    break

    consumer_thread = threading.Thread(
        target=_print_consumer,
        args=(pipeline.output_queue_ref,),
        daemon=True,
        name="keyframe-consumer",
    )
    consumer_thread.start()

    try:
        pipeline.run()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
        pipeline.stop()


if __name__ == "__main__":
    main()
