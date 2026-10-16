from __future__ import annotations

__version__ = "0.30.9-pruned"

# Annotators for bounding boxes, labels, and tracking traces
from supervision.annotators.core import (
    BackgroundOverlayAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    CircleAnnotator,
    ColorAnnotator,
    DotAnnotator,
    EllipseAnnotator,
    HaloAnnotator,
    HeatMapAnnotator,
    IconAnnotator,
    LabelAnnotator,
    OrientedBoxAnnotator,
    PercentageBarAnnotator,
    RichLabelAnnotator,
    RoundBoxAnnotator,
    TraceAnnotator,
    TriangleAnnotator,
)
from supervision.annotators.utils import (
    ColorLookup,
    hex_to_rgba,
    is_valid_hex,
    rgba_to_hex,
)

# Core Object Detection
from supervision.detection.core import Detections
from supervision.detection.line_zone import (
    LineZone,
    LineZoneAnnotator,
    LineZoneAnnotatorMulticlass,
)
from supervision.detection.tools.csv_sink import CSVSink
from supervision.detection.tools.inference_slicer import InferenceSlicer
from supervision.detection.tools.json_sink import JSONSink
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.detection.tools.smoother import DetectionsSmoother
from supervision.detection.utils.boxes import (
    clip_boxes,
    denormalize_boxes,
    move_boxes,
    pad_boxes,
    scale_boxes,
    xyxyxyxy_to_xyxy,
)
from supervision.detection.utils.converters import (
    xcycwh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_xcycarh,
    xyxy_to_xywh,
)
from supervision.detection.utils.iou_and_nms import (
    OverlapFilter,
    OverlapMetric,
    box_iou,
    box_iou_batch,
    box_non_max_merge,
    box_non_max_suppression,
    box_soft_non_max_suppression,
)
from supervision.detection.utils.matching import match_detections

# ByteTrack Multi-Object Tracker
from supervision.tracker.byte_tracker.core import ByteTrack

# Drawing and Styling
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import (
    calculate_optimal_line_thickness,
    calculate_optimal_text_scale,
    draw_filled_polygon,
    draw_filled_rectangle,
    draw_image,
    draw_line,
    draw_polygon,
    draw_rectangle,
    draw_text,
)

# Geometry
from supervision.geometry.core import CoordinatesTransformation, Point, Position, Rect
from supervision.geometry.utils import get_polygon_center

# Video and Image Utilities
from supervision.utils.conversion import cv2_to_pillow, pillow_to_cv2
from supervision.utils.file import list_files_with_extensions
from supervision.utils.image import (
    ImageSink,
    crop_image,
    get_image_resolution_wh,
    grayscale_image,
    letterbox_image,
    load_image_from_url,
    resize_image,
    scale_image,
    tint_image,
)
from supervision.utils.video import (
    FPSMonitor,
    VideoInfo,
    VideoSink,
    get_video_frames_generator,
    process_video,
)

__all__ = [
    # Core detection & tracking
    "Detections",
    "ByteTrack",
    "LineZone",
    "LineZoneAnnotator",
    "LineZoneAnnotatorMulticlass",
    "PolygonZone",
    "PolygonZoneAnnotator",
    "DetectionsSmoother",
    "InferenceSlicer",
    "CSVSink",
    "JSONSink",
    # Annotators
    "BoxAnnotator",
    "BoxCornerAnnotator",
    "RoundBoxAnnotator",
    "LabelAnnotator",
    "RichLabelAnnotator",
    "TraceAnnotator",
    "ColorAnnotator",
    "DotAnnotator",
    "CircleAnnotator",
    "TriangleAnnotator",
    "PercentageBarAnnotator",
    "HeatMapAnnotator",
    "IconAnnotator",
    "BackgroundOverlayAnnotator",
    "ColorLookup",
    # Drawing & Geometry
    "Color",
    "ColorPalette",
    "Point",
    "Position",
    "Rect",
    "CoordinatesTransformation",
    # Box utilities
    "box_iou",
    "box_iou_batch",
    "box_non_max_suppression",
    "box_non_max_merge",
    "clip_boxes",
    "denormalize_boxes",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "match_detections",
    # Video utilities
    "VideoInfo",
    "VideoSink",
    "get_video_frames_generator",
    "process_video",
    "FPSMonitor",
    "ImageSink",
]
