"""
steering_estimation.py
----------------------
Computes a steering angle estimate from lane detection output.

Approach:
    - Uses the lane centre deviation from the frame centre
      to derive a proportional steering angle.
    - Applies a PID-style correction with configurable gains
      so the output behaves like a real steering controller,
      not just a raw geometry calculation.
    - All values are in degrees. Negative = turn left, Positive = turn right.
    - Output is clamped to the physical steering range defined in config.

No neural network.  Pure geometry + control theory.
This is the same conceptual approach used in production LKAS (Lane Keep
Assist System) ECUs before the path-prediction layer.
"""

import time
from dataclasses import dataclass, field
from typing import Deque, Optional
from collections import deque

import numpy as np

from utils import (
    CameraConfig,
    FPSCounter,
    FrameResult,
    draw_hud_text,
    draw_steering_indicator,
    get_logger,
)
from lane_detection import LaneDetector, LaneConfig, LaneResult, render_lanes

import cv2

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SteeringConfig:
    """
    Parameters for the steering estimation controller.

    Gains
    -----
    kp  : Proportional gain — direct response to current lane offset.
    ki  : Integral gain     — corrects sustained drift (e.g. road camber).
    kd  : Derivative gain   — damps oscillation by reacting to rate of change.

    Tuning guide
    ------------
    Start with kp=0.6, ki=0.0, kd=0.1.
    Increase kp if the car drifts.
    Increase kd if the correction oscillates.
    Add ki only if a persistent offset remains after kp/kd tuning.
    """
    # PID gains
    kp: float = 0.6
    ki: float = 0.02
    kd: float = 0.15

    # Physical steering limits (degrees)
    max_steering_angle: float = 45.0
    min_steering_angle: float = -45.0

    # Integral windup guard — clamp accumulated error to this range
    integral_clamp: float = 30.0

    # Smoothing — EMA weight applied to final angle output
    output_ema_alpha: float = 0.30

    # Dead-band — angles below this threshold are treated as straight ahead
    dead_band_deg: float = 0.5

    # History window for derivative calculation (frames)
    derivative_window: int = 5

    # Camera mount offset — positive shifts reference point right (metres)
    # Set to 0.0 for a centre-mounted camera
    camera_lateral_offset_m: float = 0.0

    # Pixels per metre estimate at the base of the ROI (used for physical units)
    pixels_per_metre: float = 400.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _deviation_to_angle(
    deviation_px: float,
    frame_height: int,
    pixels_per_metre: float,
) -> float:
    """
    Convert lateral pixel deviation at the base of the frame to a
    steering angle in degrees.

    Uses the small-angle approximation:
        angle = arctan(lateral_error / lookahead_distance)

    lookahead_distance is estimated as half the visible road length,
    approximated from frame height and pixels_per_metre.

    Parameters
    ----------
    deviation_px     : signed pixel distance from lane centre to frame centre
    frame_height     : frame height in pixels
    pixels_per_metre : calibrated pixel density at the base of the ROI

    Returns
    -------
    Steering angle in degrees. Negative = left, Positive = right.
    """
    if pixels_per_metre <= 0:
        return 0.0

    lateral_error_m    = deviation_px / pixels_per_metre
    lookahead_distance = (frame_height * 0.55) / pixels_per_metre  # ~55% of frame
    lookahead_distance = max(lookahead_distance, 0.5)               # floor at 0.5 m

    angle_rad = np.arctan2(lateral_error_m, lookahead_distance)
    return float(np.degrees(angle_rad))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------

class PIDController:
    """
    Discrete-time PID controller operating on lane deviation error.

    Error convention:
        error > 0  -> vehicle is to the right of lane centre -> steer left (negative angle)
        error < 0  -> vehicle is to the left of lane centre  -> steer right (positive angle)
    """

    def __init__(self, config: SteeringConfig) -> None:
        self.cfg             = config
        self._integral       = 0.0
        self._prev_error     = 0.0
        self._prev_time      = time.perf_counter()
        self._error_history: Deque[float] = deque(maxlen=config.derivative_window)

    def compute(self, error: float) -> float:
        """
        Compute PID output for the given error value.

        Parameters
        ----------
        error : current lane deviation in degrees (positive = vehicle right of centre)

        Returns
        -------
        Correction angle in degrees.
        """
        now = time.perf_counter()
        dt  = now - self._prev_time
        dt  = max(dt, 1e-4)   # guard against division by zero on first call

        # Proportional
        proportional = self.cfg.kp * error

        # Integral with windup guard
        self._integral += error * dt
        self._integral  = _clamp(
            self._integral,
            -self.cfg.integral_clamp,
            self.cfg.integral_clamp,
        )
        integral = self.cfg.ki * self._integral

        # Derivative — computed over a rolling window to reduce noise
        self._error_history.append(error)
        if len(self._error_history) >= 2:
            d_error = (self._error_history[-1] - self._error_history[0]) / (
                dt * len(self._error_history)
            )
        else:
            d_error = (error - self._prev_error) / dt

        derivative = self.cfg.kd * d_error

        # PID sum — note sign: positive error -> steer left -> negative output
        output = -(proportional + integral + derivative)

        self._prev_error = error
        self._prev_time  = now

        return output

    def reset(self) -> None:
        """Reset integrator and history — call on source switch or re-init."""
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = time.perf_counter()
        self._error_history.clear()


# ---------------------------------------------------------------------------
# SteeringEstimator — main public class
# ---------------------------------------------------------------------------

class SteeringEstimator:
    """
    Computes a steering angle from a LaneResult.

    Wraps the PID controller, applies dead-band, output smoothing,
    and physical clamping.

    Usage
    -----
        estimator = SteeringEstimator(steering_config, camera_config)
        angle     = estimator.compute(lane_result)
    """

    def __init__(
        self,
        steering_config: Optional[SteeringConfig] = None,
        camera_config:   Optional[CameraConfig]   = None,
    ) -> None:
        self.cfg      = steering_config or SteeringConfig()
        self.cam_cfg  = camera_config   or CameraConfig()
        self._pid     = PIDController(self.cfg)
        self._prev_angle = 0.0

        logger.info(
            "SteeringEstimator initialised | kp=%.3f ki=%.3f kd=%.3f | "
            "limits=[%.1f, %.1f] deg",
            self.cfg.kp, self.cfg.ki, self.cfg.kd,
            self.cfg.min_steering_angle, self.cfg.max_steering_angle,
        )

    def compute(self, lane_result: LaneResult) -> float:
        """
        Derive a steering angle from the given LaneResult.

        Parameters
        ----------
        lane_result : output of LaneDetector.process()

        Returns
        -------
        Steering angle in degrees.
        Negative = turn left.  Positive = turn right.
        0.0 returned when lane is not detected.
        """
        if not lane_result.lane_detected or lane_result.centre_x is None:
            logger.debug("Lane not detected — holding previous steering angle.")
            return self._prev_angle

        # Pixel deviation: positive = lane centre is right of frame centre
        deviation_px = (
            lane_result.centre_x
            - lane_result.frame_centre_x
            + self._lateral_offset_px()
        )

        # Convert deviation to a geometry-based angle
        raw_angle = _deviation_to_angle(
            deviation_px,
            self.cam_cfg.frame_height,
            self.cfg.pixels_per_metre,
        )

        # PID correction on top of geometric angle
        pid_correction = self._pid.compute(raw_angle)
        corrected_angle = raw_angle + pid_correction

        # Dead-band
        if abs(corrected_angle) < self.cfg.dead_band_deg:
            corrected_angle = 0.0

        # Output EMA smoothing
        smoothed = (
            self.cfg.output_ema_alpha * corrected_angle
            + (1 - self.cfg.output_ema_alpha) * self._prev_angle
        )

        # Physical clamp
        final_angle = _clamp(
            smoothed,
            self.cfg.min_steering_angle,
            self.cfg.max_steering_angle,
        )

        self._prev_angle = final_angle

        logger.debug(
            "dev_px=%+.1f raw_ang=%+.2f pid=%+.2f final=%+.2f deg",
            deviation_px, raw_angle, pid_correction, final_angle,
        )

        return final_angle

    def reset(self) -> None:
        """Reset PID state and smoothing history."""
        self._pid.reset()
        self._prev_angle = 0.0

    def _lateral_offset_px(self) -> float:
        """
        Convert camera lateral mount offset from metres to pixels.
        Positive offset shifts the reference point rightward.
        """
        return self.cfg.camera_lateral_offset_m * self.cfg.pixels_per_metre


# ---------------------------------------------------------------------------
# Telemetry container
# ---------------------------------------------------------------------------

@dataclass
class SteeringTelemetry:
    """
    Per-frame steering telemetry record.
    Used for logging, replay, and digital twin feed.
    """
    timestamp_ms:    float
    deviation_px:    float
    raw_angle_deg:   float
    final_angle_deg: float
    pid_integral:    float
    lane_detected:   bool


def build_telemetry(
    estimator: SteeringEstimator,
    lane_result: LaneResult,
    final_angle: float,
) -> SteeringTelemetry:
    """Construct a telemetry snapshot from current estimator state."""
    deviation_px = 0.0
    raw_angle    = 0.0

    if lane_result.lane_detected and lane_result.centre_x is not None:
        deviation_px = lane_result.centre_x - lane_result.frame_centre_x
        raw_angle    = _deviation_to_angle(
            deviation_px,
            estimator.cam_cfg.frame_height,
            estimator.cfg.pixels_per_metre,
        )

    return SteeringTelemetry(
        timestamp_ms=time.perf_counter() * 1000,
        deviation_px=deviation_px,
        raw_angle_deg=raw_angle,
        final_angle_deg=final_angle,
        pid_integral=estimator._pid._integral,
        lane_detected=lane_result.lane_detected,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_steering(
    frame: np.ndarray,
    angle_deg: float,
    telemetry: Optional[SteeringTelemetry] = None,
) -> None:
    """
    Draw steering HUD elements onto frame.
    Mutates frame in-place.

    Renders:
        - Steering bar at bottom of frame
        - Angle readout
        - Optional telemetry overlay (deviation, PID integral)
    """
    h, w = frame.shape[:2]

    draw_steering_indicator(frame, angle_deg, w, h)

    if telemetry is not None:
        draw_hud_text(
            frame,
            f"Dev: {telemetry.deviation_px:+.1f}px  "
            f"Raw: {telemetry.raw_angle_deg:+.2f}d  "
            f"I: {telemetry.pid_integral:+.2f}",
            (12, h - 55),
            scale=0.45,
            colour_key="text_primary",
        )


# ---------------------------------------------------------------------------
# Standalone test entry point
# ---------------------------------------------------------------------------

def _run_on_video(
    source: str,
    cam_cfg:  CameraConfig,
    lane_cfg: LaneConfig,
    steer_cfg: SteeringConfig,
) -> None:
    import os
    if not os.path.exists(source):
        raise FileNotFoundError(f"Video not found: {source}")

    cap       = cv2.VideoCapture(source)
    detector  = LaneDetector(cam_cfg, lane_cfg)
    estimator = SteeringEstimator(steer_cfg, cam_cfg)
    fps_ctr   = FPSCounter()

    logger.info("Running steering estimation on: %s", source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lane_result = detector.process(frame)
        angle       = estimator.compute(lane_result)
        telemetry   = build_telemetry(estimator, lane_result, angle)

        render_lanes(frame, lane_result, lane_cfg)
        render_steering(frame, angle, telemetry)

        fps_ctr.tick()
        fps_ctr.render(frame)

        cv2.imshow("ADAS — Steering Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADAS steering estimation standalone test")
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--kp",     type=float, default=0.6)
    parser.add_argument("--ki",     type=float, default=0.02)
    parser.add_argument("--kd",     type=float, default=0.15)
    args = parser.parse_args()

    cam_cfg   = CameraConfig(frame_width=args.width, frame_height=args.height)
    lane_cfg  = LaneConfig()
    steer_cfg = SteeringConfig(kp=args.kp, ki=args.ki, kd=args.kd)

    _run_on_video(args.source, cam_cfg, lane_cfg, steer_cfg)