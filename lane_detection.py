"""
lane_detection.py
-----------------
Classical computer vision lane detection pipeline.

Pipeline per frame:
    raw frame
    -> undistort
    -> grayscale + gaussian blur
    -> canny edge detection
    -> ROI mask (trapezoid)
    -> probabilistic hough transform
    -> line averaging (left / right lane)
    -> lane centre + departure check
    -> annotated frame output

No neural network is used here.  Classical CV is faster, deterministic,
and sufficient for structured road lane detection on an ECU.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    CameraConfig,
    FPSCounter,
    FrameResult,
    apply_gaussian_blur,
    apply_roi_mask,
    draw_filled_polygon,
    draw_hud_text,
    draw_line,
    draw_steering_indicator,
    get_logger,
    lane_roi_vertices,
    to_grayscale,
    undistort_frame,
    COLOUR,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LaneConfig:
    """
    Tunable parameters for the lane detection pipeline.
    Adjust canny thresholds and hough parameters for different lighting
    and road conditions.
    """
    # Gaussian blur
    blur_kernel: int = 5

    # Canny edge detection
    canny_low:  int = 50
    canny_high: int = 150

    # Probabilistic Hough transform
    hough_rho:          float = 1.0       # distance resolution (pixels)
    hough_theta:        float = np.pi / 180  # angle resolution (radians)
    hough_threshold:    int   = 40        # minimum votes
    hough_min_line_len: int   = 60        # minimum line length in pixels
    hough_max_line_gap: int   = 25        # maximum gap between line segments

    # Slope filter — reject near-horizontal and near-vertical segments
    slope_min_abs: float = 0.4
    slope_max_abs: float = 2.5

    # Lane line draw parameters
    line_thickness: int = 4

    # Departure threshold: fraction of frame width from centre
    departure_threshold: float = 0.08

    # Temporal smoothing — exponential moving average weight for new frame
    ema_alpha: float = 0.25


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class LaneLine:
    """Represents one averaged lane boundary (left or right)."""
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    slope: float = 0.0
    intercept: float = 0.0
    valid: bool = False


@dataclass
class LaneResult:
    """Output of a single-frame lane detection pass."""
    left:           LaneLine = field(default_factory=LaneLine)
    right:          LaneLine = field(default_factory=LaneLine)
    centre_x:       Optional[int] = None   # pixel x of lane midpoint
    frame_centre_x: int = 0
    departure:      float = 0.0            # signed offset from centre, normalised
    lane_detected:  bool = False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _canny_edges(
    frame: np.ndarray,
    config: LaneConfig,
) -> np.ndarray:
    """Convert to grayscale, blur, then apply Canny edge detector."""
    gray    = to_grayscale(frame)
    blurred = apply_gaussian_blur(gray, config.blur_kernel)
    edges   = cv2.Canny(blurred, config.canny_low, config.canny_high)
    return edges


def _hough_lines(
    edges: np.ndarray,
    config: LaneConfig,
) -> Optional[np.ndarray]:
    """Run probabilistic Hough transform on an edge image."""
    lines = cv2.HoughLinesP(
        edges,
        rho=config.hough_rho,
        theta=config.hough_theta,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_len,
        maxLineGap=config.hough_max_line_gap,
    )
    return lines


def _separate_lines(
    lines: np.ndarray,
    config: LaneConfig,
    frame_centre_x: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Split raw Hough segments into left-lane and right-lane groups
    based on slope sign and position relative to frame centre.
    Returns (left_params, right_params) where each param is (slope, intercept).
    """
    left_params:  List[Tuple[float, float]] = []
    right_params: List[Tuple[float, float]] = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # skip perfectly vertical segments

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < config.slope_min_abs or abs(slope) > config.slope_max_abs:
            continue  # reject near-horizontal and near-vertical noise

        # Left lane: negative slope (in image coords), line on left half
        if slope < 0 and x1 < frame_centre_x and x2 < frame_centre_x:
            left_params.append((slope, intercept))
        # Right lane: positive slope, line on right half
        elif slope > 0 and x1 > frame_centre_x and x2 > frame_centre_x:
            right_params.append((slope, intercept))

    return left_params, right_params


def _average_lane(
    params: List[Tuple[float, float]],
    frame_height: int,
    roi_top_y: int,
) -> LaneLine:
    """
    Average a list of (slope, intercept) pairs into a single LaneLine
    that spans from the bottom of the frame to the top of the ROI.
    """
    if not params:
        return LaneLine(valid=False)

    slopes     = [p[0] for p in params]
    intercepts = [p[1] for p in params]

    slope     = float(np.mean(slopes))
    intercept = float(np.mean(intercepts))

    # Solve x = (y - b) / m for bottom and top y values
    y1 = frame_height
    y2 = roi_top_y

    if abs(slope) < 1e-6:
        return LaneLine(valid=False)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return LaneLine(
        x1=x1, y1=y1,
        x2=x2, y2=y2,
        slope=slope,
        intercept=intercept,
        valid=True,
    )


def _smooth_line(
    previous: LaneLine,
    current: LaneLine,
    alpha: float,
) -> LaneLine:
    """
    Exponential moving average between the previous and current lane line.
    Falls back to current when previous is invalid.
    Falls back to previous when current is invalid.
    """
    if not current.valid:
        return previous
    if not previous.valid:
        return current

    def ema(prev: float, curr: float) -> int:
        return int(alpha * curr + (1 - alpha) * prev)

    return LaneLine(
        x1=ema(previous.x1, current.x1),
        y1=ema(previous.y1, current.y1),
        x2=ema(previous.x2, current.x2),
        y2=ema(previous.y2, current.y2),
        slope=alpha * current.slope + (1 - alpha) * previous.slope,
        intercept=alpha * current.intercept + (1 - alpha) * previous.intercept,
        valid=True,
    )


# ---------------------------------------------------------------------------
# LaneDetector — stateful, handles smoothing across frames
# ---------------------------------------------------------------------------

class LaneDetector:
    """
    Stateful lane detector.  Maintains temporal smoothing state
    across frames.  Designed to be instantiated once and called
    every frame inside the main pipeline loop.

    Usage
    -----
        detector = LaneDetector(camera_config, lane_config)
        result   = detector.process(frame)
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        lane_config:   Optional[LaneConfig] = None,
    ) -> None:
        self.cam_cfg  = camera_config
        self.cfg      = lane_config or LaneConfig()
        self._prev_left  = LaneLine()
        self._prev_right = LaneLine()
        logger.info(
            "LaneDetector initialised | resolution=%dx%d",
            camera_config.frame_width,
            camera_config.frame_height,
        )

    def process(self, frame: np.ndarray) -> LaneResult:
        """
        Run full lane detection on a single BGR frame.
        Returns a LaneResult with left/right lanes and departure info.
        """
        h, w = frame.shape[:2]
        roi_vertices = lane_roi_vertices(w, h)
        roi_top_y    = int(h * 0.45)

        # 1. Undistort
        undistorted = undistort_frame(frame, self.cam_cfg)

        # 2. Edges
        edges = _canny_edges(undistorted, self.cfg)

        # 3. ROI mask
        masked = apply_roi_mask(edges, roi_vertices)

        # 4. Hough
        lines = _hough_lines(masked, self.cfg)

        if lines is None:
            logger.debug("No Hough lines detected in frame.")
            return LaneResult(
                left=self._prev_left,
                right=self._prev_right,
                frame_centre_x=w // 2,
                lane_detected=False,
            )

        # 5. Separate left / right
        left_params, right_params = _separate_lines(lines, self.cfg, w // 2)

        # 6. Average into single lines
        raw_left  = _average_lane(left_params,  h, roi_top_y)
        raw_right = _average_lane(right_params, h, roi_top_y)

        # 7. Temporal smoothing
        smooth_left  = _smooth_line(self._prev_left,  raw_left,  self.cfg.ema_alpha)
        smooth_right = _smooth_line(self._prev_right, raw_right, self.cfg.ema_alpha)

        self._prev_left  = smooth_left
        self._prev_right = smooth_right

        # 8. Lane centre and departure
        lane_detected = smooth_left.valid and smooth_right.valid
        centre_x      = None
        departure     = 0.0

        if lane_detected:
            # Midpoint between the two lanes at the bottom of the frame
            centre_x  = (smooth_left.x1 + smooth_right.x1) // 2
            frame_mid = w // 2
            departure = (centre_x - frame_mid) / (w / 2)   # range -1..+1

        return LaneResult(
            left=smooth_left,
            right=smooth_right,
            centre_x=centre_x,
            frame_centre_x=w // 2,
            departure=departure,
            lane_detected=lane_detected,
        )

    def reset(self) -> None:
        """Clear smoothing state — call when switching video sources."""
        self._prev_left  = LaneLine()
        self._prev_right = LaneLine()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_lanes(
    frame: np.ndarray,
    result: LaneResult,
    config: Optional[LaneConfig] = None,
) -> None:
    """
    Draw lane lines and the filled lane corridor onto frame.
    Mutates frame in-place.
    """
    cfg = config or LaneConfig()
    h, w = frame.shape[:2]

    if result.left.valid and result.right.valid:
        # Filled corridor polygon
        corridor = np.array([
            [result.left.x1,  result.left.y1],
            [result.left.x2,  result.left.y2],
            [result.right.x2, result.right.y2],
            [result.right.x1, result.right.y1],
        ], dtype=np.int32)
        draw_filled_polygon(frame, corridor, colour_key="lane_centre", alpha=0.25)

        # Lane boundary lines
        draw_line(
            frame,
            result.left.x1,  result.left.y1,
            result.left.x2,  result.left.y2,
            colour_key="lane_left",
            thickness=cfg.line_thickness,
        )
        draw_line(
            frame,
            result.right.x1, result.right.y1,
            result.right.x2, result.right.y2,
            colour_key="lane_right",
            thickness=cfg.line_thickness,
        )

        # Centre marker
        if result.centre_x is not None:
            cv2.circle(frame, (result.centre_x, h - 20), 6, COLOUR["lane_centre"], -1)

    # Departure warning
    if result.lane_detected:
        dep = result.departure
        if abs(dep) > (config or LaneConfig()).departure_threshold:
            direction = "LEFT" if dep < 0 else "RIGHT"
            draw_hud_text(
                frame,
                f"LANE DEPARTURE — {direction}",
                (w // 2 - 160, 60),
                scale=0.75,
                colour_key="text_warning",
                thickness=2,
            )
    else:
        draw_hud_text(
            frame, "LANE: NOT DETECTED",
            (12, 60), scale=0.6, colour_key="text_warning",
        )


# ---------------------------------------------------------------------------
# Standalone test entry point
# ---------------------------------------------------------------------------

def _run_on_video(source: str, cam_cfg: CameraConfig, lane_cfg: LaneConfig) -> None:
    import os
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video not found: {source}")

    cap      = cv2.VideoCapture(source)
    detector = LaneDetector(cam_cfg, lane_cfg)
    fps_ctr  = FPSCounter()

    logger.info("Running lane detection on: %s", source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process(frame)
        render_lanes(frame, result, lane_cfg)
        fps_ctr.tick()
        fps_ctr.render(frame)

        dep_pct = result.departure * 100
        draw_hud_text(frame, f"Departure: {dep_pct:+.1f}%", (12, 90), scale=0.55)

        cv2.imshow("ADAS — Lane Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADAS lane detection standalone test")
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    cam_cfg  = CameraConfig(frame_width=args.width, frame_height=args.height)
    lane_cfg = LaneConfig()

    _run_on_video(args.source, cam_cfg, lane_cfg)