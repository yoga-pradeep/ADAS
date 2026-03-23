"""
main.py
-------
ADAS pipeline orchestrator.

Wires together:
    - LaneDetector         (lane_detection.py)
    - SteeringEstimator    (steering_estimation.py)
    - BlindSpotMonitor     (blind_spot.py)

Runtime loop:
    camera / video source
    -> frame capture
    -> undistort
    -> lane detection
    -> steering estimation
    -> blind spot inference
    -> HUD rendering
    -> display / optional video write
    -> telemetry log

Design principles:
    - Each module is independently replaceable.
    - No module imports another module directly — all wired here.
    - Config objects are built once at startup and passed down.
    - The loop is single-threaded for prototype clarity.
      In production, inference and capture would run on separate threads.
    - Graceful shutdown on 'q' keypress or end of video stream.
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils import (
    CameraConfig,
    FrameResult,
    FPSCounter,
    draw_hud_text,
    get_logger,
    open_camera,
    open_video,
    undistort_frame,
    COLOUR,
)
from lane_detection import (
    LaneConfig,
    LaneDetector,
    render_lanes,
)
from steering_estimation import (
    SteeringConfig,
    SteeringEstimator,
    SteeringTelemetry,
    build_telemetry,
    render_steering,
)
from blind_spot import (
    BlindSpotConfig,
    BlindSpotMonitor,
    BlindSpotResult,
    render_blind_spot,
)

logger = get_logger(__name__, level=logging.INFO)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Top-level configuration for the full ADAS pipeline.
    Aggregates all sub-module configs plus runtime options.
    """

    # Source — either a device index (int) or a file path (str)
    source: str = "0"

    # Sub-module configs
    camera:   CameraConfig   = field(default_factory=CameraConfig)
    lane:     LaneConfig     = field(default_factory=LaneConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    blind:    BlindSpotConfig = field(default_factory=BlindSpotConfig)

    # Runtime
    show_display:   bool = True
    write_output:   bool = False
    output_path:    str  = "output/adas_output.mp4"
    log_telemetry:  bool = True
    telemetry_path: str  = "output/telemetry.csv"

    # Display
    display_width:  int = 1280
    display_height: int = 720

    # Module toggles — disable for benchmarking individual components
    enable_lane:        bool = True
    enable_steering:    bool = True
    enable_blind_spot:  bool = True


# ---------------------------------------------------------------------------
# Telemetry CSV writer
# ---------------------------------------------------------------------------

class TelemetryWriter:
    """
    Writes per-frame telemetry to a CSV file.
    Used for offline analysis, PID tuning, and digital twin replay.
    """

    HEADER = (
        "timestamp_ms,fps,steering_angle_deg,deviation_px,"
        "raw_angle_deg,pid_integral,lane_detected,"
        "blind_left,blind_right,inference_ms\n"
    )

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", buffering=1)
        self._file.write(self.HEADER)
        logger.info("Telemetry CSV: %s", path)

    def write(
        self,
        result:    FrameResult,
        telemetry: Optional[SteeringTelemetry],
        bs_result: Optional[BlindSpotResult],
    ) -> None:
        deviation_px  = telemetry.deviation_px    if telemetry else 0.0
        raw_angle     = telemetry.raw_angle_deg   if telemetry else 0.0
        pid_integral  = telemetry.pid_integral    if telemetry else 0.0
        inference_ms  = bs_result.inference_time_ms if bs_result else 0.0

        row = (
            f"{result.timestamp_ms:.2f},"
            f"{result.fps:.2f},"
            f"{result.steering_angle:.4f},"
            f"{deviation_px:.2f},"
            f"{raw_angle:.4f},"
            f"{pid_integral:.4f},"
            f"{int(result.lane_detected)},"
            f"{int(result.blind_spot_left)},"
            f"{int(result.blind_spot_right)},"
            f"{inference_ms:.2f}\n"
        )
        self._file.write(row)

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def _build_video_writer(
    path: str,
    width: int,
    height: int,
    fps: float = 30.0,
) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer at: {path}")
    logger.info("Video writer opened: %s", path)
    return writer


# ---------------------------------------------------------------------------
# HUD rendering
# ---------------------------------------------------------------------------

def _render_hud(
    frame:      np.ndarray,
    result:     FrameResult,
    telemetry:  Optional[SteeringTelemetry],
    bs_result:  Optional[BlindSpotResult],
    fps_ctr:    FPSCounter,
) -> None:
    """
    Render all HUD elements onto the frame.
    Layering order: blind spot -> lane -> steering -> status bar.
    """
    h, w = frame.shape[:2]

    # Status bar background strip at top
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)

    # FPS
    fps_ctr.render(frame, position=(8, 22))

    # Lane status
    lane_status  = "LANE: OK" if result.lane_detected else "LANE: LOST"
    lane_colour  = "text_primary" if result.lane_detected else "text_warning"
    draw_hud_text(frame, lane_status, (160, 22), scale=0.55, colour_key=lane_colour)

    # Steering angle
    draw_hud_text(
        frame,
        f"STEER: {result.steering_angle:+.1f}d",
        (320, 22),
        scale=0.55,
        colour_key="text_primary",
    )

    # Blind spot status
    bs_status = []
    if result.blind_spot_left:
        bs_status.append("BS-L")
    if result.blind_spot_right:
        bs_status.append("BS-R")

    bs_text   = "  ".join(bs_status) if bs_status else "BS: CLEAR"
    bs_colour = "text_warning" if bs_status else "text_primary"
    draw_hud_text(frame, bs_text, (530, 22), scale=0.55, colour_key=bs_colour)

    # Timestamp
    ts_s = result.timestamp_ms / 1000.0
    draw_hud_text(
        frame,
        f"{ts_s:.2f}s",
        (w - 90, 22),
        scale=0.45,
        colour_key="text_primary",
    )


# ---------------------------------------------------------------------------
# Source builder
# ---------------------------------------------------------------------------

def _open_source(cfg: PipelineConfig) -> cv2.VideoCapture:
    """
    Open a camera device or video file based on the source string.
    An integer string (e.g. "0") is treated as a device index.
    """
    source = cfg.source.strip()
    if source.isdigit():
        logger.info("Opening camera device: %s", source)
        cfg.camera.device_index = int(source)
        return open_camera(cfg.camera)
    else:
        logger.info("Opening video file: %s", source)
        return open_video(source)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class ADASPipeline:
    """
    Full ADAS prototype pipeline.

    Instantiate once, call run() to start the processing loop.
    All modules are initialised in __init__ so startup errors surface
    before the loop begins.

    Usage
    -----
        cfg      = PipelineConfig(source="data/samples/road.mp4")
        pipeline = ADASPipeline(cfg)
        pipeline.run()
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

        logger.info("Initialising ADAS pipeline...")

        self._cap = _open_source(cfg)

        # Read actual resolution from source
        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        logger.info("Source resolution: %dx%d @ %.1f fps", src_w, src_h, src_fps)

        # Update camera config to match actual source resolution
        cfg.camera.frame_width  = src_w
        cfg.camera.frame_height = src_h

        # Modules
        self._lane_detector  = LaneDetector(cfg.camera, cfg.lane)   if cfg.enable_lane       else None
        self._steer_estimator= SteeringEstimator(cfg.steering, cfg.camera) if cfg.enable_steering else None
        self._blind_monitor  = BlindSpotMonitor(cfg.blind, cfg.camera)     if cfg.enable_blind_spot else None

        # FPS counter
        self._fps_ctr = FPSCounter(window=30)

        # Optional outputs
        self._writer: Optional[cv2.VideoWriter] = None
        if cfg.write_output:
            self._writer = _build_video_writer(
                cfg.output_path, src_w, src_h, src_fps
            )

        self._tel_writer: Optional[TelemetryWriter] = None
        if cfg.log_telemetry:
            self._tel_writer = TelemetryWriter(cfg.telemetry_path)

        self._frame_count = 0
        logger.info("Pipeline ready.")

    def run(self) -> None:
        """Start the main processing loop. Blocks until source ends or 'q' pressed."""
        logger.info("Pipeline loop started. Press 'q' to quit.")

        try:
            while True:
                ret, raw_frame = self._cap.read()
                if not ret:
                    logger.info("End of source stream.")
                    break

                frame  = undistort_frame(raw_frame, self.cfg.camera)
                result = self._process_frame(frame)

                if self.cfg.show_display:
                    display = cv2.resize(
                        frame,
                        (self.cfg.display_width, self.cfg.display_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    cv2.imshow("ADAS Prototype", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Quit signal received.")
                        break

                if self._writer is not None:
                    self._writer.write(frame)

                self._frame_count += 1

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down.")
        finally:
            self._shutdown()

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Run all enabled modules on a single frame.
        Renders results onto frame in-place.
        Returns a populated FrameResult.
        """
        self._fps_ctr.tick()

        result = FrameResult(
            frame=frame,
            fps=self._fps_ctr.fps,
            timestamp_ms=time.perf_counter() * 1000,
        )

        lane_result = None
        telemetry   = None
        bs_result   = None

        # 1. Blind spot — run first so bounding boxes render under lane lines
        if self._blind_monitor is not None:
            bs_result              = self._blind_monitor.process(frame)
            result.blind_spot_left  = bs_result.alert_left
            result.blind_spot_right = bs_result.alert_right
            result.detections       = bs_result.detections
            render_blind_spot(frame, bs_result)

        # 2. Lane detection
        if self._lane_detector is not None:
            lane_result          = self._lane_detector.process(frame)
            result.lane_detected = lane_result.lane_detected
            render_lanes(frame, lane_result, self.cfg.lane)

        # 3. Steering estimation — depends on lane result
        if self._steer_estimator is not None and lane_result is not None:
            angle                = self._steer_estimator.compute(lane_result)
            result.steering_angle = angle
            telemetry            = build_telemetry(self._steer_estimator, lane_result, angle)
            render_steering(frame, angle, telemetry)

        # 4. HUD overlay (top status bar + FPS)
        _render_hud(frame, result, telemetry, bs_result, self._fps_ctr)

        # 5. Telemetry log
        if self._tel_writer is not None:
            self._tel_writer.write(result, telemetry, bs_result)

        return result

    def _shutdown(self) -> None:
        """Release all resources cleanly."""
        logger.info(
            "Shutdown | frames_processed=%d avg_fps=%.1f",
            self._frame_count,
            self._fps_ctr.fps,
        )
        self._cap.release()

        if self._writer is not None:
            self._writer.release()
            logger.info("Video output written: %s", self.cfg.output_path)

        if self._tel_writer is not None:
            self._tel_writer.close()
            logger.info("Telemetry written: %s", self.cfg.telemetry_path)

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ADAS Prototype — Lane Detection, Steering Estimation, Blind Spot Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    parser.add_argument(
        "--source", type=str, default="0",
        help="Camera device index (e.g. 0) or path to video file",
    )

    # Camera
    parser.add_argument("--width",  type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720,  help="Frame height")

    # Model
    parser.add_argument(
        "--model", type=str, default="models/yolov8n.pt",
        help="Path to YOLOv8 weights file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Inference device: cpu | mps | cuda",
    )
    parser.add_argument(
        "--conf", type=float, default=0.40,
        help="YOLOv8 confidence threshold",
    )

    # PID gains
    parser.add_argument("--kp", type=float, default=0.6,  help="Steering PID proportional gain")
    parser.add_argument("--ki", type=float, default=0.02, help="Steering PID integral gain")
    parser.add_argument("--kd", type=float, default=0.15, help="Steering PID derivative gain")

    # Output
    parser.add_argument("--save-video",    action="store_true", help="Write annotated video output")
    parser.add_argument("--output-path",   type=str, default="output/adas_output.mp4")
    parser.add_argument("--no-telemetry",  action="store_true", help="Disable telemetry CSV logging")
    parser.add_argument("--no-display",    action="store_true", help="Disable live display window")

    # Module toggles
    parser.add_argument("--no-lane",       action="store_true", help="Disable lane detection")
    parser.add_argument("--no-steering",   action="store_true", help="Disable steering estimation")
    parser.add_argument("--no-blind-spot", action="store_true", help="Disable blind spot monitor")

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    cfg = PipelineConfig(
        source=args.source,

        camera=CameraConfig(
            frame_width=args.width,
            frame_height=args.height,
        ),

        lane=LaneConfig(),

        steering=SteeringConfig(
            kp=args.kp,
            ki=args.ki,
            kd=args.kd,
        ),

        blind=BlindSpotConfig(
            model_path=args.model,
            confidence_threshold=args.conf,
            inference_device=args.device,
        ),

        show_display=not args.no_display,
        write_output=args.save_video,
        output_path=args.output_path,
        log_telemetry=not args.no_telemetry,

        enable_lane=not args.no_lane,
        enable_steering=not args.no_steering,
        enable_blind_spot=not args.no_blind_spot,
    )

    logger.info("Starting ADAS pipeline | source=%s device=%s", cfg.source, cfg.blind.inference_device)

    pipeline = ADASPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()