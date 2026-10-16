from __future__ import annotations

"""
edge/tracking/yolov8_bytetrack.py
================================
Edge-side video analytics pipeline combining YOLOv8 object detection
with ByteTrack multi-object tracking using pruned supervision components.
"""

from typing import Iterator, Optional, Tuple
import numpy as np

import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack


class YOLOv8ByteTrackPipeline:
    """
    Edge inference pipeline integrating YOLOv8 detection with ByteTrack tracking.
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.tracker = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def process_frame(
        self, frame: np.ndarray, annotate: bool = True
    ) -> Tuple[sv.Detections, Optional[np.ndarray]]:
        """
        Run YOLOv8 inference and update ByteTrack tracks for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame from OpenCV or VideoCapture.
        annotate : bool
            Whether to return an annotated frame.

        Returns
        -------
        detections : sv.Detections
            Detections object with active tracker_id values.
        annotated_frame : Optional[np.ndarray]
            Annotated frame if annotate=True, else None.
        """
        # YOLOv8 inference
        results = self.model(
            frame,
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
        )[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Update ByteTrack tracker
        detections = self.tracker.update_with_detections(detections)

        annotated_frame = None
        if annotate:
            labels = [
                f"#{tracker_id} {class_name} {confidence:.2f}"
                for tracker_id, class_name, confidence in zip(
                    detections.tracker_id,
                    detections.data.get("class_name", [""] * len(detections)),
                    detections.confidence if detections.confidence is not None else [0.0] * len(detections),
                )
            ]
            annotated_frame = frame.copy()
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels=labels
            )

        return detections, annotated_frame

    def process_video(
        self,
        source_video_path: str,
        target_video_path: str,
    ) -> None:
        """
        Process an entire video file from source to target.
        """
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        frame_generator = sv.get_video_frames_generator(source_video_path)

        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in frame_generator:
                _, annotated_frame = self.process_frame(frame, annotate=True)
                if annotated_frame is not None:
                    sink.write_frame(annotated_frame)
