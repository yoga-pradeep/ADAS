"""
blind_spot.py
-------------
Real-time blind spot monitoring using YOLOv8 inference.

Architecture:
    - Loads a pretrained YOLOv8n model (COCO weights).
    - Defines left and right blind spot zones as configurable
      rectangular ROIs on the frame.
    - Runs YOLOv8 inference on the full frame.
    - Filters detections to only those whose bounding box centroid
      falls inside a blind spot zone.
    - Tracks objects across frames using a simple IoU-based tracker
      to avoid false positives from single-frame detections.
    - Raises a BlindSpotAlert when a vehicle-class object is confirmed
      in either zone for a minimum number of consecutive frames.

COCO vehicle classes used (no fine-tuning required):
    1  : bicycle
    2  : car
    3  : motorcycle
    5  : bus
    7  : truck

No dataset download or training needed.
YOLOv8n pretrained on COCO is sufficient for prototype-grade detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    CameraConfig,
    COLOUR,
    FPSCounter,
    draw_bounding_box,
    draw_hud_text,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# COCO class filter — vehicle classes only
# ---------------------------------------------------------------------------

VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

COCO_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BlindSpotConfig:
    """
    Configuration for blind spot zone geometry, detection thresholds,
    and alert logic.

    Zone geometry is expressed as fractions of frame width/height
    so the config is resolution-independent.

    Left zone  : left edge of frame  (driver side mirror region)
    Right zone : right edge of frame (passenger side mirror region)
    """

    model_path: str = "models/yolov8n.pt"

    # Inference
    confidence_threshold: float = 0.40
    iou_threshold:        float = 0.45
    inference_device:     str   = "cpu"   # "cpu" | "mps" | "cuda"

    # Left blind spot zone (fraction of frame)
    left_zone_x_start:  float = 0.0
    left_zone_x_end:    float = 0.22
    left_zone_y_start:  float = 0.30
    left_zone_y_end:    float = 0.85

    # Right blind spot zone (fraction of frame)
    right_zone_x_start: float = 0.78
    right_zone_x_end:   float = 1.0
    right_zone_y_start: float = 0.30
    right_zone_y_end:   float = 0.85

    # Alert logic
    # Object must appear in zone for this many consecutive frames
    # before an alert is raised — reduces false positives
    alert_min_frames: int   = 3
    alert_cooldown_s: float = 1.5   # minimum seconds between repeated alerts

    # Simple IoU tracker — match threshold
    tracker_iou_threshold: float = 0.35


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    """Pixel-coordinate blind spot zone rectangle."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str   # "LEFT" or "RIGHT"

    def contains_centroid(self, bx1: int, by1: int, bx2: int, by2: int) -> bool:
        """Return True if the centroid of the given box is inside this zone."""
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def overlap_fraction(self, bx1: int, by1: int, bx2: int, by2: int) -> float:
        """
        Return the fraction of the detection box that overlaps with this zone.
        Useful as a secondary check alongside centroid containment.
        """
        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        box_area = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / box_area


def build_zones(
    frame_width: int,
    frame_height: int,
    cfg: BlindSpotConfig,
) -> Tuple[Zone, Zone]:
    """Convert fractional zone config to pixel-coordinate Zone objects."""
    left_zone = Zone(
        x1=int(cfg.left_zone_x_start  * frame_width),
        y1=int(cfg.left_zone_y_start  * frame_height),
        x2=int(cfg.left_zone_x_end    * frame_width),
        y2=int(cfg.left_zone_y_end    * frame_height),
        label="LEFT",
    )
    right_zone = Zone(
        x1=int(cfg.right_zone_x_start * frame_width),
        y1=int(cfg.right_zone_y_start * frame_height),
        x2=int(cfg.right_zone_x_end   * frame_width),
        y2=int(cfg.right_zone_y_end   * frame_height),
        label="RIGHT",
    )
    return left_zone, right_zone


# ---------------------------------------------------------------------------
# Detection result container
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection result from YOLOv8."""
    x1:         int
    y1:         int
    x2:         int
    y2:         int
    confidence: float
    class_id:   int
    class_name: str
    track_id:   int = -1
    in_left:    bool = False
    in_right:   bool = False


@dataclass
class BlindSpotResult:
    """
    Output for a single processed frame.
    Consumed by main.py and the digital twin feed.
    """
    detections:        List[Detection] = field(default_factory=list)
    alert_left:        bool = False
    alert_right:       bool = False
    left_zone:         Optional[Zone] = None
    right_zone:        Optional[Zone] = None
    inference_time_ms: float = 0.0
    timestamp_ms:      float = field(default_factory=lambda: time.perf_counter() * 1000)


# ---------------------------------------------------------------------------
# IoU-based object tracker
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    track_id:      int
    x1: int; y1: int; x2: int; y2: int
    class_id:      int
    consecutive_frames: int = 1
    last_seen:     float = field(default_factory=time.perf_counter)


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Compute Intersection over Union between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / max(union, 1)


class SimpleTracker:
    """
    Lightweight IoU-based tracker.
    Assigns consistent track IDs across frames without requiring
    deep sort or Kalman filtering.
    Sufficient for a prototype-grade ADAS blind spot monitor.
    """

    def __init__(self, iou_threshold: float = 0.35, max_age_s: float = 0.5) -> None:
        self._threshold  = iou_threshold
        self._max_age    = max_age_s
        self._tracks:    Dict[int, TrackedObject] = {}
        self._next_id    = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Match incoming detections to existing tracks.
        Updates consecutive_frames count on matched tracks.
        Assigns new track IDs to unmatched detections.
        Returns detections with track_id populated.
        """
        now = time.perf_counter()

        # Expire old tracks
        expired = [
            tid for tid, t in self._tracks.items()
            if (now - t.last_seen) > self._max_age
        ]
        for tid in expired:
            del self._tracks[tid]

        matched_track_ids = set()

        for det in detections:
            det_box   = (det.x1, det.y1, det.x2, det.y2)
            best_iou  = 0.0
            best_tid  = -1

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                if track.class_id != det.class_id:
                    continue
                score = _iou(det_box, (track.x1, track.y1, track.x2, track.y2))
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_iou >= self._threshold and best_tid >= 0:
                # Matched existing track
                track = self._tracks[best_tid]
                track.x1 = det.x1; track.y1 = det.y1
                track.x2 = det.x2; track.y2 = det.y2
                track.consecutive_frames += 1
                track.last_seen = now
                det.track_id    = best_tid
                matched_track_ids.add(best_tid)
            else:
                # New track
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = TrackedObject(
                    track_id=new_id,
                    x1=det.x1, y1=det.y1,
                    x2=det.x2, y2=det.y2,
                    class_id=det.class_id,
                )
                det.track_id = new_id

        return detections

    def get_consecutive_frames(self, track_id: int) -> int:
        """Return how many consecutive frames this track has been seen."""
        if track_id in self._tracks:
            return self._tracks[track_id].consecutive_frames
        return 0

    def reset(self) -> None:
        self._tracks.clear()


# ---------------------------------------------------------------------------
# BlindSpotMonitor — main public class
# ---------------------------------------------------------------------------

class BlindSpotMonitor:
    """
    Stateful blind spot monitor.

    Loads YOLOv8n on initialisation.
    Call process(frame) every frame in the pipeline loop.

    Usage
    -----
        monitor = BlindSpotMonitor(config, camera_config)
        result  = monitor.process(frame)
    """

    def __init__(
        self,
        config:        Optional[BlindSpotConfig] = None,
        camera_config: Optional[CameraConfig]    = None,
    ) -> None:
        self.cfg     = config        or BlindSpotConfig()
        self.cam_cfg = camera_config or CameraConfig()

        self._model   = self._load_model()
        self._tracker = SimpleTracker(
            iou_threshold=self.cfg.tracker_iou_threshold
        )

        self._left_alert_ts  = 0.0
        self._right_alert_ts = 0.0
        self._zones: Optional[Tuple[Zone, Zone]] = None

        logger.info(
            "BlindSpotMonitor initialised | model=%s device=%s conf=%.2f",
            self.cfg.model_path,
            self.cfg.inference_device,
            self.cfg.confidence_threshold,
        )

    def _load_model(self):
        """
        Load YOLOv8 model via the ultralytics API.
        Raises FileNotFoundError if the weights file is missing.
        """
        from ultralytics import YOLO

        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at: {model_path}\n"
                f"Run: curl -L https://github.com/ultralytics/assets/releases/"
                f"download/v8.1.0/yolov8n.pt -o {model_path}"
            )

        model = YOLO(str(model_path))
        logger.info("YOLOv8 model loaded: %s", model_path)
        return model

    def _get_zones(self, frame_height: int, frame_width: int) -> Tuple[Zone, Zone]:
        """Build and cache zones at the first frame's resolution."""
        if self._zones is None:
            self._zones = build_zones(frame_width, frame_height, self.cfg)
            logger.info(
                "Blind spot zones built | LEFT=%s RIGHT=%s",
                self._zones[0], self._zones[1],
            )
        return self._zones

    def process(self, frame: np.ndarray) -> BlindSpotResult:
        """
        Run YOLOv8 inference and blind spot analysis on a single BGR frame.

        Parameters
        ----------
        frame : BGR numpy array from camera or video

        Returns
        -------
        BlindSpotResult with detections, zone alerts, and timing.
        """
        h, w = frame.shape[:2]
        left_zone, right_zone = self._get_zones(h, w)

        t0 = time.perf_counter()

        raw_results = self._model.predict(
            source=frame,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            device=self.cfg.inference_device,
            verbose=False,
            classes=list(VEHICLE_CLASS_IDS),
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._parse_detections(raw_results)
        detections = self._tracker.update(detections)
        detections = self._assign_zones(detections, left_zone, right_zone)

        alert_left, alert_right = self._evaluate_alerts(detections)

        logger.debug(
            "Frame | dets=%d left_alert=%s right_alert=%s inf=%.1fms",
            len(detections), alert_left, alert_right, inference_ms,
        )

        return BlindSpotResult(
            detections=detections,
            alert_left=alert_left,
            alert_right=alert_right,
            left_zone=left_zone,
            right_zone=right_zone,
            inference_time_ms=inference_ms,
        )

    def _parse_detections(self, raw_results) -> List[Detection]:
        """Convert ultralytics result objects to Detection dataclasses."""
        detections: List[Detection] = []

        for result in raw_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf            = float(box.conf[0].item())

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=COCO_NAMES.get(cls_id, str(cls_id)),
                ))

        return detections

    def _assign_zones(
        self,
        detections: List[Detection],
        left_zone:  Zone,
        right_zone: Zone,
    ) -> List[Detection]:
        """Tag each detection with which blind spot zone it occupies."""
        for det in detections:
            det.in_left  = left_zone.contains_centroid(det.x1, det.y1, det.x2, det.y2)
            det.in_right = right_zone.contains_centroid(det.x1, det.y1, det.x2, det.y2)
        return detections

    def _evaluate_alerts(
        self,
        detections: List[Detection],
    ) -> Tuple[bool, bool]:
        """
        Raise alerts only for objects that have been tracked for
        at least alert_min_frames consecutive frames and respect
        the cooldown period.
        """
        now          = time.perf_counter()
        alert_left   = False
        alert_right  = False

        for det in detections:
            consec = self._tracker.get_consecutive_frames(det.track_id)
            if consec < self.cfg.alert_min_frames:
                continue

            if det.in_left:
                if (now - self._left_alert_ts) > self.cfg.alert_cooldown_s:
                    alert_left          = True
                    self._left_alert_ts = now

            if det.in_right:
                if (now - self._right_alert_ts) > self.cfg.alert_cooldown_s:
                    alert_right          = True
                    self._right_alert_ts = now

        return alert_left, alert_right

    def reset(self) -> None:
        """Reset tracker state — call when switching video sources."""
        self._tracker.reset()
        self._zones = None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_blind_spot(
    frame: np.ndarray,
    result: BlindSpotResult,
) -> None:
    """
    Draw blind spot zones, detections, and alert overlays onto frame.
    Mutates frame in-place.
    """
    h, w = frame.shape[:2]

    # Draw zone rectangles
    for zone in [result.left_zone, result.right_zone]:
        if zone is None:
            continue
        is_alert = (zone.label == "LEFT" and result.alert_left) or \
                   (zone.label == "RIGHT" and result.alert_right)

        colour = COLOUR["blind_left"] if is_alert else (80, 80, 80)
        alpha  = 0.35 if is_alert else 0.10

        overlay = frame.copy()
        cv2.rectangle(overlay, (zone.x1, zone.y1), (zone.x2, zone.y2), colour, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), colour, 2)

        label_y = zone.y1 - 8 if zone.y1 > 20 else zone.y1 + 16
        draw_hud_text(
            frame,
            f"BS {zone.label}",
            (zone.x1 + 4, label_y),
            scale=0.5,
            colour_key="text_primary",
        )

    # Draw detections
    for det in result.detections:
        colour_key = "blind_left" if (det.in_left or det.in_right) else "detection_box"
        draw_bounding_box(
            frame,
            det.x1, det.y1, det.x2, det.y2,
            label=det.class_name,
            confidence=det.confidence,
            colour_key=colour_key,
        )

    # Alert banners
    if result.alert_left:
        draw_hud_text(
            frame,
            "WARNING: VEHICLE IN LEFT BLIND SPOT",
            (12, h - 90),
            scale=0.65,
            colour_key="text_warning",
            thickness=2,
        )

    if result.alert_right:
        draw_hud_text(
            frame,
            "WARNING: VEHICLE IN RIGHT BLIND SPOT",
            (w - 420, h - 90),
            scale=0.65,
            colour_key="text_warning",
            thickness=2,
        )

    # Inference time
    draw_hud_text(
        frame,
        f"Inference: {result.inference_time_ms:.1f}ms",
        (12, h - 120),
        scale=0.45,
        colour_key="text_primary",
    )


# ---------------------------------------------------------------------------
# Standalone test entry point
# ---------------------------------------------------------------------------

def _run_on_video(
    source: str,
    cam_cfg: CameraConfig,
    bs_cfg:  BlindSpotConfig,
) -> None:
    import os
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video not found: {source}")

    cap     = cv2.VideoCapture(source)
    monitor = BlindSpotMonitor(bs_cfg, cam_cfg)
    fps_ctr = FPSCounter()

    logger.info("Running blind spot monitor on: %s", source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = monitor.process(frame)
        render_blind_spot(frame, result)

        fps_ctr.tick()
        fps_ctr.render(frame)

        cv2.imshow("ADAS — Blind Spot Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADAS blind spot monitor standalone test")
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--model",  type=str, default="models/yolov8n.pt")
    parser.add_argument("--conf",   type=float, default=0.40)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    cam_cfg = CameraConfig(frame_width=args.width, frame_height=args.height)
    bs_cfg  = BlindSpotConfig(
        model_path=args.model,
        confidence_threshold=args.conf,
        inference_device=args.device,
    )

    _run_on_video(args.source, cam_cfg, bs_cfg)