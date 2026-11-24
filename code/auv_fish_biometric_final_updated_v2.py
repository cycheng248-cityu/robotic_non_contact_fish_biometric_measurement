#!/usr/bin/env python3
"""30-second video acquisition followed by strided biometric inference; 180s target.
Default stationary/passive. No heading sensor or physical calibration is invented.
Powered repetition requires fresh independently verified measured heading.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import signal
import socket
import threading
import time
import traceback
import subprocess
import multiprocessing as mp
import select
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional
import cv2
import numpy as np

CAMERAS = {"underwater_left": "/dev/video0", "underwater_right": "/dev/video2", "top": "/dev/video4"}
CAMERA_RESOLUTIONS = {"underwater_left": (1280, 720), "underwater_right": (1280, 720), "top": (640, 480)}
CAMERA_FPS = {"underwater_left": 25, "underwater_right": 25, "top": 25}
MODEL_PATH = "/home/cycheng248/auv_project/models/S3_1280_SGD_cosine_250_lr001/weights/best_ncnn_model_1280"
OUTPUT_ROOT = "/home/cycheng248/auv_project/results/end-to-end_test"
TOP_X1_PIXEL, TOP_X2_PIXEL = 30.0, 625.0
TOP_Y1_PIXEL, TOP_Y2_PIXEL = 440.0, 75.0
TANK_LENGTH_CM, TANK_WIDTH_CM = 40.0, 27.0
TRACK_X_MIN_CM, TRACK_X_MAX_CM = 0.0, 30.0
TRACK_Y_MIN_CM, TRACK_Y_MAX_CM = 0.0, 27.0
AUV_X_CM, AUV_Y_CM, AUV_INITIAL_HEADING_DEG = 30.0, 13.0, 180.0
AUV_MASK_CENTER_PIXEL, AUV_MASK_RADIUS_PIXEL = (500, 257), 60
LEFT_SCALE_COEFFS = np.array([-1.3369e-05, 1.2516e-03, -5.6337e-03], dtype=float)
RIGHT_SCALE_COEFFS = np.array([-1.3369e-05, 1.2516e-03, -5.6337e-03], dtype=float)
CALIBRATION_RANGE_MIN_CM, CALIBRATION_RANGE_MAX_CM = 11.670, 30.000
CLASS_FISH_SIDE, CLASS_FISH_TOP = 0, 1
HEAD, TAIL_BASE, UPPER, LOWER = 0, 1, 2, 3
YOLO_IMAGE_SIZE = 1280
TOP_DIAGNOSTIC_CONF, TOP_MEASUREMENT_CONF = 0.10, 0.30
SIDE_DIAGNOSTIC_CONF, SIDE_MEASUREMENT_CONF = 0.01, 0.18
KEYPOINT_MEASUREMENT_CONF = 0.50
YOLO_IOU, YOLO_MAX_DETECTIONS, TOP_ASSOCIATION_MAX_PX = 0.45, 10, 100.0
ENABLE_SIDE_ENHANCEMENT = True
CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE = 2.0, (8, 8)
GAMMA_DARK, GAMMA_NORMAL = 0.72, 0.88
DARK_MEDIAN_THRESHOLD, NORMAL_MEDIAN_THRESHOLD = 70.0, 115.0
ENABLE_MILD_SHARPEN = False
PROCESS_SCALE = 0.50
MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_LEARNING_RATE = 120, 26, 0.005
WARMUP_FRAMES, TANK_EDGE_MARGIN_PX, DARK_GRAY_MAX = 5, 5, 100
MIN_CANDIDATE_AREA, MAX_CANDIDATE_AREA = 20.0, 1800.0
MIN_ASPECT_RATIO, MAX_ASPECT_RATIO, MIN_SOLIDITY = 1.0, 8.5, 0.25
MORPH_OPEN, MORPH_CLOSE = 3, 7
EMA_ALPHA, MAX_MISSED_FRAMES, MAX_CONTOUR_JUMP_PX = 0.35, 4, 75.0  # jump at processing resolution
MAX_TRACK_SPEED_CM_S = 10.0
MIN_STABLE_FRAMES, MIN_TRACK_AGE_SECONDS = 5, 0.0
MAX_TRIGGER_SPEED_CM_S, TRIGGER_WINDOW = 4.0, 10
MAX_TRIGGER_POSITION_STD_CM, MEASUREMENT_COOLDOWN_SECONDS = 0.8, 3.0
BURST_FRAME_COUNT, BURST_INTERVAL_SECONDS = 30, 0.10
SELECTED_TRIPLETS, MAX_TRIPLET_SKEW_SECONDS = 5, 0.080
CAPTURE_SETTLE_SECONDS = 0.50
MIN_STANDARD_LENGTH_CM, MAX_STANDARD_LENGTH_CM = 1.0, 30.0
MIN_BODY_HEIGHT_CM, MAX_BODY_HEIGHT_CM = 0.2, 15.0
ESP32_IP, ESP32_PORT = "ESP32_IP", 4210
MOTOR_SEND_INTERVAL_SECONDS, YAW_MAGNITUDE = 0.20, 0.15
YAW_DEADBAND_DEG, YAW_HYSTERESIS_DEG = 8.0, 4.0
YAW_PULSE_MIN_SECONDS, YAW_PULSE_MAX_SECONDS = 0.05, 1.00
YAW_SETTLE_SECONDS, YAW_RATE_DEG_PER_SECOND = 0.80, math.nan
ACK_FRESHNESS_SECONDS = 10.00
MAX_ORIENT_ATTEMPTS = 100
THERMAL_CHECK_INTERVAL_SECONDS, STOP_TEMPERATURE_C = 2.0, 78.0
DIFF_LAG = 4
DIFF_THRESHOLD = 15
CURRENT_DARK_MAX = 150
TOP_MARGIN_SEMANTICS_VERIFIED = False
TOP_PLANE_CALIBRATION_VERIFIED = False
TOP_WIDTH_POINT_INDICES = (UPPER, LOWER)  # ONLY usable after label verification

@dataclass
class SimpleTrackResult:
    timestamp: float
    valid: bool = False
    state: str = "WARMUP"
    reason: str = ""
    u_px: float = math.nan
    v_px: float = math.nan
    x_cm: float = math.nan
    y_cm: float = math.nan
    range_cm: float = math.nan
    bearing_error_deg: float = math.nan
    speed_cm_s: float = math.nan
    stable_frames: int = 0
    track_age_s: float = 0.0
    candidate_area: float = math.nan
    candidate_aspect: float = math.nan
    candidate_solidity: float = math.nan
    position_std_cm: float = math.inf
    scale_range_valid: bool = False
    trigger_ready: bool = False


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def csv_value(value: Any, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else ""


def normalize_angle(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def top_pixel_to_tank_cm(u_px: float, v_px: float) -> tuple[float, float]:
    x_cm = (float(u_px) - TOP_X1_PIXEL) * TANK_LENGTH_CM / (TOP_X2_PIXEL - TOP_X1_PIXEL)
    y_cm = (TOP_Y1_PIXEL - float(v_px)) * TANK_WIDTH_CM / (TOP_Y1_PIXEL - TOP_Y2_PIXEL)
    return float(x_cm), float(y_cm)


def tank_cm_to_top_pixel(x_cm: float, y_cm: float) -> tuple[float, float]:
    return (TOP_X1_PIXEL + x_cm * (TOP_X2_PIXEL - TOP_X1_PIXEL) / TANK_LENGTH_CM,
            TOP_Y1_PIXEL - y_cm * (TOP_Y1_PIXEL - TOP_Y2_PIXEL) / TANK_WIDTH_CM)


def inside_tracking_area(x_cm: float, y_cm: float) -> bool:
    return (finite(x_cm) and finite(y_cm) and TRACK_X_MIN_CM <= x_cm <= TRACK_X_MAX_CM
            and TRACK_Y_MIN_CM <= y_cm <= TRACK_Y_MAX_CM)


def physical_range_cm(x_cm: float, y_cm: float) -> float:
    # Retains original one-dimensional independent variable; verify against calibration experiment.
    return abs(AUV_X_CM - float(x_cm))


def bearing_error_deg(x_cm: float, y_cm: float, heading_deg: float) -> float:
    return normalize_angle(math.degrees(math.atan2(y_cm - AUV_Y_CM, x_cm - AUV_X_CM)) - heading_deg)


def read_pi_status() -> tuple[float, str]:
    try:
        raw = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, stderr=subprocess.DEVNULL)
        temperature = float(raw.split("=")[1].split("'")[0])
    except Exception:
        temperature = math.nan
    try:
        throttled = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        throttled = "unavailable"
    return temperature, throttled


def safe_imwrite(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save {path}")


def open_camera(role: str, device: str) -> cv2.VideoCapture:
    width, height = CAMERA_RESOLUTIONS[role]
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open {role} camera at {device}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS[role])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            cap.grab()
        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (width, height):
            raise RuntimeError(f"{role}: got {actual}, expected {(width, height)}")
        return cap
    except Exception:
        cap.release()
        raise


def read_camera(cap: cv2.VideoCapture, role: str) -> tuple[np.ndarray, float]:
    ok, frame = cap.read()
    timestamp = time.time()  # Acquisition completion time, not hardware exposure time.
    if not ok or frame is None:
        raise RuntimeError(f"Camera read failed: {role}")
    return frame, timestamp


class SimpleFishTracker:
    def __init__(self) -> None:
        self.proc_w = round(640 * PROCESS_SCALE)
        self.proc_h = round(480 * PROCESS_SCALE)
        self.allowed_mask = self.make_allowed_mask()
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN, MORPH_OPEN))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE, MORPH_CLOSE))
        self.reset_track(warmup=True)

    def make_allowed_mask(self) -> np.ndarray:
        mask = np.zeros((self.proc_h, self.proc_w), np.uint8)
        u1, v1 = tank_cm_to_top_pixel(TRACK_X_MIN_CM, TRACK_Y_MAX_CM)
        u2, v2 = tank_cm_to_top_pixel(TRACK_X_MAX_CM, TRACK_Y_MIN_CM)
        margin = TANK_EDGE_MARGIN_PX
        cv2.rectangle(mask, (round(u1 * PROCESS_SCALE) + margin, round(v1 * PROCESS_SCALE) + margin),
                      (round(u2 * PROCESS_SCALE) - margin, round(v2 * PROCESS_SCALE) - margin), 255, -1)
        center = tuple(round(n * PROCESS_SCALE) for n in AUV_MASK_CENTER_PIXEL)
        cv2.circle(mask, center, round(AUV_MASK_RADIUS_PIXEL * PROCESS_SCALE), 0, -1)
        return mask

    def reset_track(self, warmup: bool = True) -> None:
        self.gray_history = deque(maxlen=DIFF_LAG + 1)
        self.last_mask = np.zeros((self.proc_h, self.proc_w), np.uint8)
        self.state = "WARMUP" if warmup else "SEARCH"
        self.warmup_count = 0
        self.missed_frames = 0
        self.stable_frames = 0
        self.ema_u: Optional[float] = None
        self.ema_v: Optional[float] = None
        self.ema_speed = 0.0
        self.track_start = 0.0
        self.last_measurement_time = 0.0
        self.prev_x_cm: Optional[float] = None
        self.prev_y_cm: Optional[float] = None
        self.position_history: deque[tuple[float, float]] = deque(maxlen=TRIGGER_WINDOW)

    def make_foreground_mask(self, gray: np.ndarray, learning_rate: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if len(self.gray_history) < DIFF_LAG:
            self.gray_history.append(blurred.copy())
            self.last_mask.fill(0)
            return self.last_mask.copy()
        previous = self.gray_history[-DIFF_LAG]
        self.gray_history.append(blurred.copy())
        difference = cv2.absdiff(blurred, previous)
        _, motion = cv2.threshold(difference, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion = cv2.bitwise_and(motion, self.allowed_mask)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, self.open_kernel)
        motion = cv2.dilate(motion, self.close_kernel, iterations=1)
        # Current-image support avoids reporting the entire old/new difference centroid.
        current_dark = cv2.inRange(blurred, 0, CURRENT_DARK_MAX)
        combined = cv2.bitwise_and(motion, current_dark)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.close_kernel)
        self.last_mask = cv2.bitwise_and(combined, self.allowed_mask)
        return self.last_mask.copy()

    def candidates(self, mask: np.ndarray) -> list[dict[str, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not MIN_CANDIDATE_AREA <= area <= MAX_CANDIDATE_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(1, min(w, h))
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(1e-6, hull_area)
            if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO and solidity >= MIN_SOLIDITY):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            u, v = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            if not 0 <= round(v) < self.proc_h or not 0 <= round(u) < self.proc_w:
                continue
            if self.allowed_mask[round(v), round(u)] == 0:
                continue
            candidates.append({"u": float(u), "v": float(v), "area": area,
                               "aspect": float(aspect), "solidity": solidity})
        return candidates

    def miss(self, result: SimpleTrackResult, reason: str) -> SimpleTrackResult:
        self.missed_frames += 1
        self.stable_frames = 0
        self.position_history.clear()
        self.prev_x_cm = self.prev_y_cm = None
        self.last_measurement_time = 0.0
        self.ema_speed = 0.0
        if self.missed_frames > MAX_MISSED_FRAMES:
            self.ema_u = self.ema_v = None
            self.track_start = 0.0
            self.state = "SEARCH"
            reason = "track_lost"
        result.state, result.reason = self.state, reason
        return result

    def update(self, frame: np.ndarray, heading_deg: float) -> SimpleTrackResult:
        now = time.monotonic()
        result = SimpleTrackResult(timestamp=time.time(), state=self.state)
        if frame.shape[:2] != (480, 640):
            raise RuntimeError(f"Top camera frame shape {frame.shape[:2]} != (480, 640)")
        proc = cv2.resize(frame, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        if self.state == "WARMUP":
            self.make_foreground_mask(gray, -1.0)
            self.warmup_count += 1
            if self.warmup_count >= WARMUP_FRAMES:
                self.state = "SEARCH"
            result.state, result.reason = self.state, "background_warmup"
            return result
        candidates = self.candidates(self.make_foreground_mask(gray, MOG2_LEARNING_RATE))
        if not candidates:
            return self.miss(result, "no_contours")
        if self.ema_u is None:
            candidate = max(candidates, key=lambda c: c["area"])
        else:
            close = [c for c in candidates if math.hypot(c["u"] - self.ema_u,
                                                          c["v"] - self.ema_v) <= MAX_CONTOUR_JUMP_PX]
            if not close:
                return self.miss(result, "large_jump_rejected")
            candidate = min(close, key=lambda c: math.hypot(c["u"] - self.ema_u, c["v"] - self.ema_v))
        measured_x, measured_y = top_pixel_to_tank_cm(candidate["u"] / PROCESS_SCALE,
                                                        candidate["v"] / PROCESS_SCALE)
        if not inside_tracking_area(measured_x, measured_y):
            return self.miss(result, "outside_tracking_area")
        if self.ema_u is None:
            self.ema_u, self.ema_v = candidate["u"], candidate["v"]
            self.track_start = now
        else:
            self.ema_u = EMA_ALPHA * candidate["u"] + (1 - EMA_ALPHA) * self.ema_u
            self.ema_v = EMA_ALPHA * candidate["v"] + (1 - EMA_ALPHA) * self.ema_v
        self.missed_frames = 0
        self.stable_frames += 1
        self.state = "TRACKING"
        u_full, v_full = self.ema_u / PROCESS_SCALE, self.ema_v / PROCESS_SCALE
        x_cm, y_cm = top_pixel_to_tank_cm(u_full, v_full)
        dt = now - self.last_measurement_time if self.last_measurement_time else 0.0
        if dt > 0 and self.prev_x_cm is not None and self.prev_y_cm is not None:
            speed = math.hypot(x_cm - self.prev_x_cm, y_cm - self.prev_y_cm) / dt
            self.ema_speed = EMA_ALPHA * speed + (1 - EMA_ALPHA) * self.ema_speed
        self.prev_x_cm, self.prev_y_cm, self.last_measurement_time = x_cm, y_cm, now
        self.position_history.append((x_cm, y_cm))
        position_std = math.inf
        if len(self.position_history) == TRIGGER_WINDOW:
            points = np.asarray(self.position_history, dtype=float)
            position_std = max(float(np.std(points[:, 0])), float(np.std(points[:, 1])))
        range_cm = physical_range_cm(x_cm, y_cm)
        scale_valid = CALIBRATION_RANGE_MIN_CM <= range_cm <= CALIBRATION_RANGE_MAX_CM
        valid = self.ema_speed <= MAX_TRACK_SPEED_CM_S
        age = now - self.track_start
        trigger_ready = (valid and self.stable_frames >= MIN_STABLE_FRAMES
                         and age >= MIN_TRACK_AGE_SECONDS)  # Acquisition trigger, NOT measurement approval
        return SimpleTrackResult(timestamp=time.time(), valid=valid, state=self.state,
                                 reason="measured" if valid else "speed_exceeds_track_limit",
                                 u_px=u_full, v_px=v_full, x_cm=x_cm, y_cm=y_cm,
                                 range_cm=range_cm,
                                 bearing_error_deg=bearing_error_deg(x_cm, y_cm, heading_deg),
                                 speed_cm_s=self.ema_speed, stable_frames=self.stable_frames,
                                 track_age_s=age, candidate_area=candidate["area"],
                                 candidate_aspect=candidate["aspect"],
                                 candidate_solidity=candidate["solidity"],
                                 position_std_cm=position_std, scale_range_valid=scale_valid,
                                 trigger_ready=trigger_ready)

    def annotate(self, frame: np.ndarray, estimate: SimpleTrackResult, state: str) -> np.ndarray:
        image = frame.copy()
        cv2.rectangle(image, (round(TOP_X1_PIXEL), round(TOP_Y2_PIXEL)),
                      (round(TOP_X2_PIXEL), round(TOP_Y1_PIXEL)), (0, 180, 0), 2)
        u1, v1 = tank_cm_to_top_pixel(TRACK_X_MIN_CM, TRACK_Y_MAX_CM)
        u2, v2 = tank_cm_to_top_pixel(TRACK_X_MAX_CM, TRACK_Y_MIN_CM)
        cv2.rectangle(image, (round(u1), round(v1)), (round(u2), round(v2)), (0, 165, 255), 2)
        cv2.circle(image, AUV_MASK_CENTER_PIXEL, AUV_MASK_RADIUS_PIXEL, (128, 128, 128), 1)
        if estimate.valid:
            center = (round(estimate.u_px), round(estimate.v_px))
            cv2.circle(image, center, 7, (0, 0, 255), -1)
            cv2.line(image, AUV_MASK_CENTER_PIXEL, center, (255, 255, 0), 1)
        lines = [f"Controller: {state}",
                 f"Tracker: {estimate.state} valid={int(estimate.valid)} trigger={int(estimate.trigger_ready)}",
                 f"XY cm: {estimate.x_cm:.2f}, {estimate.y_cm:.2f}" if estimate.valid else "XY cm: --",
                 f"range={estimate.range_cm:.2f} bearing={estimate.bearing_error_deg:.1f}" if estimate.valid else "range/bearing: --",
                 f"speed={estimate.speed_cm_s:.2f} stable={estimate.stable_frames}" if estimate.valid else estimate.reason]
        for index, line in enumerate(lines):
            cv2.putText(image, line, (8, 20 + index * 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 255, 255), 1, cv2.LINE_AA)
        return image


class DeadlineMotor:
    """Independent nonblocking UDP sender with sequence/value-matched ACKs.
    Software timing is best-effort. ESP32 timeout and physical stop are required.
    """
    def __init__(self,enabled,stop_event,pulse_seconds,heartbeat=0.20,endpoint=None,log_path=None,
                 allow_repeat=False,max_total_on=1.0):
        self.enabled=enabled;self.stop_event=stop_event;self.pulse_seconds=pulse_seconds
        self.heartbeat=heartbeat;self.endpoint=endpoint or (ESP32_IP,ESP32_PORT)
        self.allow_repeat=allow_repeat;self.max_total_on=max_total_on;self.reserved_on=0.0
        self.lock=threading.Lock();self.shutdown=threading.Event();self.failed=threading.Event()
        self.sock=None;self.last_reply=None;self.reply_count=0;self.error=None
        self.pending={};self.last_zero_ack=None;self.pulse_count=0
        self.recording_open=False;self.recording_deadline=0.0
        self.recording_pulse_limit=0;self.recording_start_count=0
        self.last_reject='';self.recording_stop_reason='not_started' 
        self.log_file=open(log_path,'w',newline='') if log_path else None
        self.log_writer=csv.writer(self.log_file) if self.log_file else None
        if self.log_writer:
            self.log_writer.writerow(['wall','monotonic','kind','sequence','yaw','reason','peer','payload']);self.log_file.flush()
        try:
            if enabled:
                self.endpoint=(socket.gethostbyname(self.endpoint[0]),self.endpoint[1])
                self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                self.sock.bind(('0.0.0.0',0));self.sock.setblocking(False)
                print(f'UDP bind={self.sock.getsockname()} peer={self.endpoint}',flush=True)
        except Exception:
            if self.sock:self.sock.close()
            if self.log_file:self.log_file.close()
            raise
        self.sequence=int(time.time()*1000)
        self.tracking_deadline=0.0;self.yaw=0.0;self.deadline=0.0;self.reason='initial_zero';self.consumed=False;self.history=[]
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def log(self,kind,payload='',peer='',seq=None):
        if self.log_writer:
            self.log_writer.writerow([time.time(),time.monotonic(),kind,self.sequence if seq is None else seq,
                self.yaw,self.reason,str(peer),payload]);self.log_file.flush()

    def fresh(self):
        return self.last_reply is not None and time.monotonic()-self.last_reply<= ACK_FRESHNESS_SECONDS

    def wait_for_link(self,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end and not self.stop_event.is_set() and not self.failed.is_set():
            with self.lock:
                if self.fresh():return True
            time.sleep(.02)
        return False

    def wait_zero(self,since,timeout=1):
        if not self.enabled:return True
        end=time.monotonic()+timeout
        while time.monotonic()<end and not self.failed.is_set():
            with self.lock:
                if self.last_zero_ack is not None and self.last_zero_ack>=since:return True
            time.sleep(.02)
        return False

    def set_tracking_freshness(self,valid,observed_at,max_age):
        with self.lock:
            self.tracking_deadline=observed_at+max_age if valid else 0.0

    def begin_recording(self,started,duration,max_pulses):
        with self.lock:
            if self.stop_event.is_set() or self.failed.is_set():return False
            self.recording_open=True
            self.recording_deadline=float(started)+float(duration)
            self.recording_pulse_limit=int(max_pulses)
            self.recording_start_count=self.pulse_count
            self.recording_stop_reason='recording_active'
            self.log('RECORDING_GATE_OPEN',f'duration={duration}; max_pulses={max_pulses}')
            return True

    def end_recording(self,reason='recording_finished'):
        with self.lock:
            self.recording_open=False;self.yaw=0.;self.reason=reason
            self.recording_stop_reason=reason
            self.log('RECORDING_GATE_CLOSED',reason)

    def usage(self):
        with self.lock:
            return {'requested_pulses':self.pulse_count,
                'recording_requested_pulses':self.pulse_count-self.recording_start_count,
                'max_pulses':self.recording_pulse_limit,
                'reserved_requested_motor_seconds':self.reserved_on,
                'max_requested_motor_seconds':self.max_total_on,
                'recording_open':self.recording_open,
                'recording_stop_reason':self.recording_stop_reason,
                'last_pulse_rejection':self.last_reject}

    def pulse(self,yaw):
        if not math.isfinite(float(yaw)) or not 0<abs(float(yaw))<=.15:return False
        with self.lock:
            now=time.monotonic()
            reject=None
            if self.stop_event.is_set() or self.failed.is_set():reject='stop_or_fault'
            elif not self.recording_open or now>=self.recording_deadline:reject='outside_recording_window'
            elif self.yaw:reject='pulse_already_active'
            elif self.pulse_count-self.recording_start_count>=self.recording_pulse_limit:reject='maximum_yaw_count_reached'
            elif now>=self.tracking_deadline:reject='tracking_stale'
            elif self.consumed and not self.allow_repeat:reject='repeat_disabled'
            elif self.enabled and not self.fresh():reject='no_fresh_matching_ack'
            elif self.reserved_on+self.pulse_seconds>self.max_total_on+1e-9:reject='total_requested_motor_budget'
            if reject:
                self.last_reject=reject;self.log('PULSE_REJECT',reject);return False
            self.reserved_on+=self.pulse_seconds;self.pulse_count+=1;self.consumed=True
            self.yaw=float(yaw)
            self.deadline=min(now+self.pulse_seconds,self.recording_deadline)
            self.reason='bounded_pulse';self.last_reject=''
            self.log('PULSE_REQUEST');return True

    def cancel(self,reason):
        with self.lock:self.yaw=0.;self.reason=reason

    def snapshot(self):
        with self.lock:return self.yaw,self.reason

    def _send_locked(self):
        now=time.monotonic();seq=self.sequence
        packet=f'{seq},0.000,{self.yaw:.3f},0.000'
        if self.sock:self.sock.sendto(packet.encode(),self.endpoint)
        self.pending[seq]=(now,self.yaw)
        self.pending={k:v for k,v in self.pending.items() if now-v[0]<=1.0}
        self.log('TX' if self.enabled else 'DRY_TX',packet,self.endpoint,seq)
        self.history.append((time.time(),now,self.yaw,self.reason,packet,self.enabled));self.sequence+=1

    def _receive(self):
        if not self.sock:return
        for _ in range(16):
            try:data,peer=self.sock.recvfrom(1024)
            except BlockingIOError:break
            text=data.decode('utf-8',errors='replace').strip()
            if peer!=self.endpoint:self.log('WRONG_PEER',text,peer);continue
            try:
                if not text.startswith('ACK:'):raise ValueError('prefix')
                fields=text[4:].split(',')
                if len(fields)!=4:raise ValueError('fields')
                seq=int(fields[0]);f,y,v=map(float,fields[1:])
                if not all(math.isfinite(z) for z in [f,y,v]):raise ValueError('nonfinite')
                if seq not in self.pending:raise ValueError('unknown/duplicate sequence')
                sent,expected=self.pending[seq];now=time.monotonic()
                if now-sent>.35:raise ValueError('delayed ACK')
                if abs(f)>.0006 or abs(v)>.0006 or abs(y-expected)>.0006:raise ValueError('echo mismatch')
                del self.pending[seq]
            except (ValueError,OverflowError) as exc:self.log('REJECTED_ACK',f'{exc}: {text}',peer);continue
            self.last_reply=now;self.reply_count+=1
            if abs(y)<.0006:self.last_zero_ack=sent
            self.log('MATCHED_ACK',text,peer,seq)

    def _run(self):
        # Receive frequently, but preserve the configured command-send interval.
        next_send=0.0
        last_sent_yaw=None
        try:
            while not self.shutdown.is_set():
                with self.lock:
                    self._receive()
                    now=time.monotonic()
                    if self.stop_event.is_set():
                        self.yaw=0.;self.reason='stop_or_trial_deadline'
                    elif self.recording_open and now>=self.recording_deadline:
                        self.recording_open=False;self.yaw=0.;self.reason='recording_deadline'
                        self.recording_stop_reason='recording_deadline'
                        self.log('RECORDING_GATE_CLOSED','recording_deadline')
                    elif self.yaw and now>=getattr(self,'tracking_deadline',float('inf')):
                        self.yaw=0.;self.reason='tracking_stale'
                    elif self.yaw and now>=self.deadline:
                        self.yaw=0.;self.reason='pulse_deadline'
                    elif self.yaw and self.enabled and not self.fresh():
                        age=None if self.last_reply is None else now-self.last_reply
                        age_text='never' if age is None else f'{age:.3f}s'
                        self.error=(f'ACK_stale: last accepted matching ACK age={age_text}; '
                                    f'peer={self.endpoint}; pending={len(self.pending)}. '
                                    'Yaw cancelled; inspect udp_link.csv.')
                        self.yaw=0.;self.reason='ACK_stale'
                        self.failed.set()
                        self.log('FAULT',self.error)
                    if now>=next_send or self.yaw!=last_sent_yaw:
                        self._send_locked()
                        last_sent_yaw=self.yaw
                        next_send=time.monotonic()+self.heartbeat
                    delay=min(.02,max(.001,next_send-time.monotonic()))
                    if self.yaw:
                        delay=min(delay,max(.001,self.deadline-time.monotonic()))
                self.shutdown.wait(delay)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.failed.set()
            with self.lock:
                self.yaw=0.;self.reason='sender_failed'
                try:self._send_locked()
                except Exception:pass


    def close(self):
        self.cancel('shutdown');self.shutdown.set();self.thread.join(timeout=1)
        if self.thread.is_alive():raise RuntimeError('Sender not stopped; use physical stop')
        try:
            with self.lock:
                for _ in range(5):
                    try:self._send_locked()
                    except Exception:pass
                    time.sleep(.02)
        finally:
            if self.sock:self.sock.close()
            if self.log_file:self.log_file.close()


def input_commands(stop_event, arm_event):
    # Separate from camera loop so q/Enter works during a blocked camera read.
    while not stop_event.is_set():
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            line = sys.stdin.readline()
            if line == "":
                stop_event.set()
                return
            command = line.strip().lower()
            if command == "arm":
                arm_event.set()
                print("Armed for ONE fish-triggered pulse.", flush=True)
            elif command in ("", "q", "stop"):
                stop_event.set()
            else:
                print("Commands: arm | q (or Enter to stop)", flush=True)
        except Exception:
            stop_event.set()
            return




# Isolated reference inference configuration, distinct from live calibration.
BOX_CONFIDENCE_THRESHOLD=0.50
IMG_SIZE=1280
IOU_THRESHOLD=0.70
MAX_DETECTIONS=10

def result_arrays(result):
    """Convert Ultralytics pose output to NumPy arrays."""

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
            np.empty((0, 0, 2), dtype=float)
        )

    boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(float)
    box_conf = result.boxes.conf.cpu().numpy().astype(float)
    class_id = result.boxes.cls.cpu().numpy().astype(int)

    if result.keypoints is None:
        kpts_xy = np.empty((len(boxes_xyxy), 0, 2), dtype=float)
    else:
        kpts_xy = result.keypoints.xy.cpu().numpy().astype(float)

    return boxes_xyxy, box_conf, class_id, kpts_xy

def find_one_high_confidence_fish(
    boxes_xyxy,
    box_conf,
    class_id,
    kpts_xy,
    expected_class_id,
    threshold=BOX_CONFIDENCE_THRESHOLD
):
    """
    Select one panel-specific fish detection.

    Rules:
    - Keep only expected class, e.g. fish_top or fish_side.
    - Keep only confidence >= threshold.
    - Require exactly ONE remaining candidate in this panel.
    - Return the highest-confidence candidate index.

    Returns:
    - index: selected index in original result arrays, or None.
    - reason: explanation string.
    """

    candidate_indices = np.where(
        (class_id == expected_class_id) &
        (box_conf >= threshold)
    )[0]

    if len(candidate_indices) == 0:
        return None, "no_high_confidence_target"

    if len(candidate_indices) > 1:
        return None, "more_than_one_high_confidence_target"

    jj = int(candidate_indices[np.argmax(box_conf[candidate_indices])])

    if kpts_xy.shape[0] <= jj:
        return None, "missing_keypoint_detection"

    if kpts_xy.shape[1] < 4:
        return None, "fewer_than_four_keypoint_slots"

    return jj, "selected"

def predict_arrays(model, panel):
    started = time.perf_counter()
    results = model.predict(source=panel, imgsz=IMG_SIZE, conf=BOX_CONFIDENCE_THRESHOLD,
                            iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS, device='cpu', rect=False, verbose=False)
    elapsed = (time.perf_counter()-started)*1000
    arrays = tuple(a.copy() for a in result_arrays(results[0]))
    del results
    return arrays, elapsed

def select_safe(arrays, expected):
    boxes, conf, cls, kp = arrays
    jj, reason = find_one_high_confidence_fish(boxes, conf, cls, kp, expected)
    if jj is not None:
        box = boxes[jj]
        if not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
            return None, 'malformed_box'
        if not np.all(np.isfinite(kp[jj, :4])):
            return None, 'nonfinite_keypoints'
    return jj, reason


POLY_A, POLY_B, POLY_C = 0.00013722, -0.00407794, 0.04285072
REFERENCE_PANEL_SIZES = {'top': (640,480), 'left': (1280,720), 'right': (1280,720)}

def reference_range_scale(top_box):
    x1,y1,x2,y2=map(float,top_box)
    x,y=top_pixel_to_tank_cm((x1+x2)/2,(y1+y2)/2)
    d=30.0-x
    scale=POLY_A*d*d+POLY_B*d+POLY_C
    return x,y,d,scale

def reference_side_cm(points,distance):
    pts=np.asarray(points,dtype=float)
    length=float(np.linalg.norm(pts[HEAD]-pts[TAIL_BASE]))
    height=float(abs(pts[UPPER,1]-pts[LOWER,1]))
    scale=POLY_A*distance**2+POLY_B*distance+POLY_C
    return length*scale,height*scale,length,height,scale

def reference_selection(result,role,image_shape,track_uv=None,association_max_px=100.):
    if result.boxes is None:return None,'no_detection'
    boxes=np.asarray(result.boxes.xyxy.cpu().numpy(),dtype=float)
    conf=np.asarray(result.boxes.conf.cpu().numpy(),dtype=float)
    classes=result.boxes.cls.cpu().numpy().astype(int)
    expected=CLASS_FISH_TOP if role=='top' else CLASS_FISH_SIDE
    ids=[]
    for j in np.flatnonzero((classes==expected)&np.isfinite(conf)&(conf>=.30)):
        b=boxes[j]
        if np.all(np.isfinite(b)) and b[2]>b[0] and b[3]>b[1]:ids.append(int(j))
    candidates=len(ids)
    if role=='top':
        eligible=[]
        for j in ids:
            c=(boxes[j,:2]+boxes[j,2:])/2
            x,y=top_pixel_to_tank_cm(*c)
            if not inside_tracking_area(x,y):continue
            if math.hypot(c[0]-AUV_MASK_CENTER_PIXEL[0],c[1]-AUV_MASK_CENTER_PIXEL[1])<=AUV_MASK_RADIUS_PIXEL:continue
            eligible.append(j)
        ids=eligible
        if not ids:return None,'no_top_box_in_tracking_roi'
        if track_uv is not None:
            ids=[j for j in ids if np.linalg.norm((boxes[j,:2]+boxes[j,2:])/2-np.asarray(track_uv))<=association_max_px]
            if not ids:return None,'no_top_box_associated_with_track'
            j=min(ids,key=lambda i:(np.linalg.norm((boxes[i,:2]+boxes[i,2:])/2-np.asarray(track_uv)),-conf[i]))
            reason='roi_nearest_track'
        else:
            j=max(ids,key=lambda i:conf[i]);reason='roi_highest_conf_no_valid_track'
    else:
        if not ids:return None,'no_side_target'
        j=max(ids,key=lambda i:conf[i]);reason='highest_confidence_side'
    item={'box':boxes[j],'confidence':float(conf[j]),'candidate_count':candidates,
          'eligible_count':len(ids),'selected_index':j}
    if role=='top':return item,reason
    item['points']=np.full((4,2),np.nan);item['keypoint_conf']=np.full(4,np.nan)
    kp=result.keypoints
    if kp is not None:
        if kp.xy is not None:
            pts=kp.xy.cpu().numpy()
            if pts.ndim==3 and j<len(pts):item['points'][:min(4,pts.shape[1])]=pts[j,:4]
        if kp.conf is not None:
            scores=kp.conf.cpu().numpy()
            if scores.ndim==2 and j<len(scores):item['keypoint_conf'][:min(4,scores.shape[1])]=scores[j,:4]
    return item,reason

def pair_valid(item,indices,width=1280,height=720,confidence=.30):
    if item is None:return False,'no_selected_side'
    p=item['points'][list(indices)];c=item['keypoint_conf'][list(indices)]
    if not np.all(np.isfinite(p)):return False,'nonfinite_or_missing_coordinates'
    if not np.all((p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)):
        return False,'coordinates_outside_image'
    if not np.all(np.isfinite(c)) or not np.all(c>=confidence):return False,'low_or_missing_keypoint_confidence'
    return True,'valid_pair'

def save_measurement_histograms(directory,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    accepted=rows
    summary={}
    for key,width in [('length_cm',.1),('height_cm',.1)]:
        groups={role:np.asarray([v[key] for v in accepted if v['camera']==role and v.get('accepted_'+key.split('_')[0],False) and finite(v.get(key))],dtype=float)
                for role in ('left','right')}
        all_values=np.concatenate(list(groups.values()))
        if len(all_values):
            lo=math.floor(float(all_values.min())/width)*width
            hi=math.ceil(float(all_values.max())/width)*width+width
            edges=np.arange(lo,hi+width/2,width)
            fig,ax=plt.subplots()
            with (directory/(key+'_histogram.csv')).open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['camera','bin_left_cm','bin_right_cm','count'])
                for role,a in groups.items():
                    counts,_=np.histogram(a,bins=edges)
                    for i,n in enumerate(counts):w.writerow([role,edges[i],edges[i+1],int(n)])
                    if len(a):ax.hist(a,bins=edges,alpha=.45,label=f'{role} n={len(a)}')
            ax.set(xlabel=key.replace('_',' '),ylabel='Accepted camera observations',
                   title='Reference-polynomial estimates: '+directory.name)
            ax.legend();fig.tight_layout();fig.savefig(directory/(key+'_histogram.png'));plt.close(fig)
        summary[key]={role:{'n':len(a),'mean':float(np.mean(a)) if len(a) else None,
            'median':float(np.median(a)) if len(a) else None,
            'p95':float(np.percentile(a,95)) if len(a) else None} for role,a in groups.items()}
    return summary

def four_keypoints_present(item,width=1280,height=720):
    """Presence-only rule; keypoint confidence is NOT an acceptance gate."""
    if item is None:return False,'no_selected_fish'
    points=np.asarray(item.get('points',[]),dtype=float)
    if points.ndim!=2 or points.shape[0]<4 or points.shape[1]!=2:
        return False,'fewer_than_four_keypoints'
    p=points[:4]
    if not np.all(np.isfinite(p)):return False,'missing_or_nonfinite_keypoints'
    if np.any(np.all(p==0,axis=1)):return False,'zero_placeholder_keypoint'
    if not np.all((p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)):
        return False,'keypoint_outside_image'
    return True,'four_keypoints_present'

def keypoint_indices_present(item,indices,width=1280,height=720):
    """Coordinate presence only: no keypoint-confidence threshold."""
    if item is None:return False,'no_selected_fish'
    points=np.asarray(item.get('points',[]),dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or points.shape[0]<=max(indices):
        return False,'missing_keypoint_slots'
    p=points[list(indices)]
    if not np.all(np.isfinite(p)):return False,'nonfinite_or_missing_coordinates'
    if np.any(np.all(p==0,axis=1)):return False,'zero_placeholder_keypoint'
    if not np.all((p[:,0]>=0)&(p[:,0]<width)&(p[:,1]>=0)&(p[:,1]<height)):
        return False,'keypoint_outside_image'
    return True,'required_coordinates_present'


def tracking_range_scale(frame):
    """Use this acquisition row's valid track; never reuse another row's position."""
    if str(frame.get('tracking_valid','')).strip().lower() not in ('true','1','yes'):
        return None,'tracking_invalid_or_missing'
    if finite(frame.get('fish_x_cm')) and finite(frame.get('fish_y_cm')):
        x,y=float(frame['fish_x_cm']),float(frame['fish_y_cm'])
    elif finite(frame.get('fish_u_px')) and finite(frame.get('fish_v_px')):
        x,y=top_pixel_to_tank_cm(float(frame['fish_u_px']),float(frame['fish_v_px']))
    else:return None,'tracking_coordinates_missing'
    if not inside_tracking_area(x,y):return None,'tracking_position_outside_roi'
    d=AUV_X_CM-x
    scale=POLY_A*d*d+POLY_B*d+POLY_C
    return (x,y,d,scale),'recorded_valid_tracking_centroid'


def _inference_worker_without_npz(event_dir,model_path,imgsz,threads,max_triplets,skew_limit,analysis_timeout_override=None,useful_policy_override=None):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(threads);cv2.setNumThreads(threads)
    directory=Path(event_dir)
    options=json.loads((directory/'analysis_options.json').read_text())
    stride=options['inference_stride'];deadline=options.get('analysis_deadline_monotonic',float('inf'))
    if analysis_timeout_override is not None:deadline=time.monotonic()+analysis_timeout_override
    policy=useful_policy_override or options.get('useful_policy','four-keypoints')
    output=directory/('physical_inference_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir()
    with (directory/'frames.csv').open(newline='') as f:all_frames=list(csv.DictReader(f))
    frames=all_frames[::stride]
    settings={'polynomial':[POLY_A,POLY_B,POLY_C],'calibration_pixel_space':[1280,720],
        'top_selection':'NO top-view YOLO; same-triplet saved tracker centroid',
        'side_selection':'highest fish-side box confidence; four coordinates present, no keypoint-confidence gate',
        'useful_policy':policy,
        'pair_policy_note':'optional measurement-pairs; length=head+tail, height=upper+lower, no keypoint-confidence gate',
        'length_policy':'useful frame + finite positive reference-equation conversion; phase/skew/fit logged only',
        'stride':stride,'requested_selected_triplets':len(frames),'options':options,
        'range':'30-fish_x_cm, axis proxy not radial distance','independent_samples':False}
    (output/'configuration.json').write_text(json.dumps(settings,indent=2))
    records=[];completed=0;stopped=False
    caps={};positions={role:-1 for role in ('left','right')}
    fields=['triplet','camera','phase','skew_s','top_reason','side_reason','side_confidence',
        'side_candidate_count','top_candidate_count','fish_x_cm','fish_y_cm',
        'reference_axis_range_cm','cm_per_native_pixel','length_px','height_px','length_cm',
        'height_cm','length_pair_valid','height_pair_valid','length_pair_reason','height_pair_reason',
        'in_calibration_fit_range','accepted_length','accepted_height','accepted',
        'quality_flags','predict_ms','useful_frame','useful_reason','physical_length_available','physical_height_available',
        'position_source','position_reason','tracking_position_available','tracking_side_delta_s',
        'four_keypoints_frame','useful_length_frame','useful_height_frame','useful_policy']
    try:
        for role in positions:
            caps[role]=cv2.VideoCapture(str(directory/(role+'.avi')))
            if not caps[role].isOpened():raise RuntimeError('Cannot open '+role+'.avi')
        model=YOLO(str(model_path),task='pose')
        with (output/'measurements.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();f.flush()
            for frame in frames:
                # Reserve time for summaries, report incomplete coverage explicitly.
                if time.monotonic()>=deadline-5:stopped=True;break
                index=int(frame['index']);selections={};reasons={};timings={};panels={}
                for role,cap in caps.items():
                    while positions[role]<index:
                        ok,image=cap.read();positions[role]+=1
                        if not ok:raise RuntimeError(f'Video decode failed: {role} index {index}')
                    expected=(640,480) if role=='top' else (1280,720)
                    if image.shape[1::-1]!=expected:raise RuntimeError(f'{role}: native calibration resolution mismatch')
                    panels[role]=image
                    begin=time.perf_counter()
                    result=model.predict(image,imgsz=imgsz,conf=.30,iou=.50,max_det=10,
                        rect=False,device='cpu',verbose=False)[0]
                    timings[role]=(time.perf_counter()-begin)*1000
                    selections[role],reasons[role]=reference_selection(result,role,image.shape)
                    if options['save_annotations']:
                        annotated=result.plot()
                        safe_imwrite(output/f'annotated_{index:06d}_{role}.jpg',annotated)
                    del result
                coordinates,position_reason=tracking_range_scale(frame)
                for role in ('left','right'):
                    item=selections[role];row={key:'' for key in fields}
                    four,four_reason=four_keypoints_present(item)
                    lpresent,lr=keypoint_indices_present(item,(HEAD,TAIL_BASE))
                    hpresent,hr=keypoint_indices_present(item,(UPPER,LOWER))
                    useful=four if policy=='four-keypoints' else lpresent or hpresent
                    useful_reason=four_reason if policy=='four-keypoints' else ('measurement_pair_present' if useful else 'no_measurement_pair')
                    lv=four if policy=='four-keypoints' else lpresent
                    hv=four if policy=='four-keypoints' else hpresent
                    row.update(triplet=index,camera=role,phase=frame['phase'],skew_s=float(frame['skew_s']),
                        top_reason='not_run_tracking_centroid',side_reason=reasons[role],predict_ms=timings[role],
                        side_confidence=item['confidence'] if item else '',
                        side_candidate_count=item['candidate_count'] if item else 0,
                        top_candidate_count=0,
                        length_pair_valid=lv,height_pair_valid=hv,length_pair_reason=lr,height_pair_reason=hr,
                        accepted_length=False,accepted_height=False,accepted=False,
                        useful_frame=useful,useful_reason=useful_reason,
                        physical_length_available=False,physical_height_available=False,
                        position_source='tracking_centroid',position_reason=position_reason,
                        tracking_position_available=coordinates is not None,
                        tracking_side_delta_s=(float(frame['t_'+role])-float(frame['t_top']))
                            if finite(frame.get('t_'+role)) and finite(frame.get('t_top')) else math.nan,
                        four_keypoints_frame=four,useful_length_frame=lpresent,useful_height_frame=hpresent,
                        useful_policy=policy)
                    # Pixel geometry is available even when centroid-based cm conversion is not.
                    if item is not None:
                        if lpresent:row['length_px']=float(np.linalg.norm(item['points'][HEAD]-item['points'][TAIL_BASE]))
                        if hpresent:row['height_px']=float(abs(item['points'][UPPER,1]-item['points'][LOWER,1]))
                    flags=[]
                    if frame['phase']!='observation_zero':flags.append('yaw_or_settling')
                    if float(frame['skew_s'])>skew_limit:flags.append('camera_skew')
                    if coordinates is None:flags.append(position_reason)
                    else:
                        x,y,d,scale=coordinates;fit=0<=d<=17
                        row.update(fish_x_cm=x,fish_y_cm=y,reference_axis_range_cm=d,
                            cm_per_native_pixel=scale,in_calibration_fit_range=fit)
                        if not fit:flags.append('outside_polynomial_fit_range')
                        if not math.isfinite(scale) or scale<=0:flags.append('invalid_scale')
                        if item is not None and math.isfinite(scale) and scale>0:
                            pts=item['points']
                            if lv:
                                lp=float(np.linalg.norm(pts[HEAD]-pts[TAIL_BASE]));row.update(length_px=lp,length_cm=lp*scale)
                            if hv:
                                hp=float(abs(pts[UPPER,1]-pts[LOWER,1]));row.update(height_px=hp,height_cm=hp*scale)
                    row['accepted_length']=bool(lv and finite(row['length_cm']) and row['length_cm']>0)
                    row['physical_length_available']=row['accepted_length']
                    row['accepted_height']=bool(hv and finite(row['height_cm']) and row['height_cm']>0)
                    row['physical_height_available']=row['accepted_height']
                    row['accepted']=row['accepted_length'] and row['accepted_height']
                    row['quality_flags']=';'.join(flags)
                    writer.writerow(row);records.append(row)
                f.flush();completed+=1
                (output/'progress.json').write_text(json.dumps({'completed_triplets':completed,'requested_triplets':len(frames)}))
                print(f'Analysed {completed}/{len(frames)} stride={stride}',flush=True)
    finally:
        for cap in caps.values():cap.release()
    accepted_dir=output/'accepted_histograms';accepted_dir.mkdir()
    raw_dir=output/'raw_histograms';raw_dir.mkdir()
    summary=save_measurement_histograms(accepted_dir,records)
    per_view={}
    for role in ('left','right'):
        view_rows=[row for row in records if row['camera']==role]
        view_dir=output/role;view_dir.mkdir()
        with (view_dir/'frames_and_lengths.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(view_rows)
        view_hist=save_measurement_histograms(view_dir,view_rows)
        count_useful=sum(row['useful_frame'] for row in view_rows)
        per_view[role]={
            'analysed_frames':len(view_rows),'useful_frames':count_useful,
            'four_keypoints_frames':sum(row['four_keypoints_frame'] for row in view_rows),
            'useful_length_frames':sum(row['useful_length_frame'] for row in view_rows),
            'useful_height_frames':sum(row['useful_height_frame'] for row in view_rows),
            'useful_fraction':count_useful/len(view_rows) if view_rows else None,
            'physical_length_count':sum(row['physical_length_available'] for row in view_rows),
            'physical_height_count':sum(row['physical_height_available'] for row in view_rows),
            'useful_without_physical_length':sum(row['useful_frame'] and not row['physical_length_available'] for row in view_rows),
            'measurement_summary':view_hist}
        (view_dir/'summary.json').write_text(json.dumps(per_view[role],indent=2))
    summary['per_view']=per_view
    summary['useful_frame_policy']=policy
    summary['position_source']='saved_tracking_centroid_no_top_yolo'
    summary['quality_flags_policy']='Diagnostic only, NOT histogram exclusion gates'
    summary['accepted_histograms_note']='Legacy folder name; now uses simplified four-keypoint policy'

    raw=[]
    for row in records:
        candidate=dict(row)
        for metric in ('length','height'):
            candidate['accepted_'+metric]=bool(finite(row.get(metric+'_cm')) and row[metric+'_cm']>0)
        raw.append(candidate)
    summary.update(raw=save_measurement_histograms(raw_dir,raw),
        saved_triplets=len(all_frames),requested_triplets=len(frames),completed_triplets=completed,model_predictions=2*completed,
        analysis_complete=completed==len(frames),stopped_for_budget=stopped,
        accepted_length_observations=sum(x['accepted_length'] for x in records),
        accepted_height_observations=sum(x['accepted_height'] for x in records),
        calibration_status='1280x720 reference-equation estimates; axis proxy remains an assumption')
    (output/'summary.json').write_text(json.dumps(summary,indent=2))


def open_side(role):
    return open_camera(role, CAMERAS[role])


def read_heading(path,max_age,initial):
    """Optional external measurement in tank coordinates. No sensor is invented.
    Publisher must atomically write wall-clock timestamp and heading_deg.
    """
    if path is None:return initial,True,'manual_initial'
    try:
        data=json.loads(Path(path).read_text())
        age=time.time()-float(data['timestamp'])
        heading=float(data['heading_deg'])
        if not math.isfinite(heading) or not math.isfinite(age) or age<0 or age>max_age:
            return math.nan,False,'stale_or_invalid_heading'
        return normalize_angle(heading),True,'measured_external'
    except Exception:return math.nan,False,'heading_unavailable'



class ZoneYawPolicy:
    """Repeat after each pulse/settle only from newly stable observations."""
    def __init__(self, center, half_band, stable_frames=3, stable_seconds=.5,
                 max_gap=1.5, magnitude=.15):
        self.center=float(center);self.half_band=float(half_band)
        self.required_frames=int(stable_frames)
        self.required_seconds=float(stable_seconds)
        self.max_gap=float(max_gap);self.magnitude=float(magnitude)
        self.armed=True
        self.reset_observations()

    def reset_observations(self):
        self.zone='invalid';self.count=0;self.since=None;self.last=None

    def update(self,v,valid,ready,now,blocked=False):
        # Camera observations during motion/settling never count toward
        # the next pulse, and no directional request is queued.
        if blocked:
            self.armed=False
            self.reset_observations()
            return 0.,'pulse_or_settling'
        if not valid or not ready or not math.isfinite(float(v)):
            self.armed=False
            self.reset_observations()
            return 0.,'invalid_or_unstable_track'
        zone=('upper' if v<self.center-self.half_band else
              'lower' if v>self.center+self.half_band else 'neutral')
        if self.last is None or now-self.last>self.max_gap or zone!=self.zone:
            self.zone=zone;self.count=1;self.since=now
        else:self.count+=1
        self.last=now
        if self.count<self.required_frames or now-self.since<self.required_seconds:
            self.armed=False
            return 0.,'zone_stabilizing'
        self.armed=True
        if zone=='neutral':return 0.,'neutral_hold'
        return (self.magnitude if zone=='upper' else -self.magnitude),'repeat_zone_pulse_eligible'

    def consume(self):
        self.armed=False
        self.reset_observations()




def capture_event(top_cap,tracker,args,motor,stop,event,heading):
    sides={};writers={};records=0;pulses=0;next_pulse=0.;last_yaw=False
    event_started=None
    try:
        for role in ['underwater_left','underwater_right']:
            sides[role] = open_side(role)
        for name,size in [('left',(1280,720)),('right',(1280,720)),('top',(640,480))]:
            writers[name]=cv2.VideoWriter(str(event/f'{name}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),args.record_fps,size)
            if not writers[name].isOpened():raise RuntimeError('MJPG AVI writer failed: '+name)
        last_status_check=0.;temp=math.nan;flags='unavailable'
        tracker.reset_track(warmup=True)
        if stop.is_set():return {'status':'stopped_before_capture','frames':0,'pulses':0}
        event_started=time.monotonic()
        if not motor.begin_recording(event_started,args.clip_seconds,args.max_pulses_per_event):
            raise RuntimeError('Cannot open recording yaw gate: stop/fault already active')
        fields=['index','top_file','left_file','right_file','t_top','t_left','t_right','skew_s','phase','sharpness',
            'elapsed_capture_s','tracking_valid','consecutive','fish_x_cm','fish_y_cm','bearing_error_deg',
            'heading_deg','heading_status','requested_yaw','sender_yaw','control_decision','temperature_c','fish_v_px','zone','zone_armed','zone_stable_count','fish_u_px','tracking_timestamp_monotonic']
        with (event/'frames.csv').open('w',newline='') as f:
            out=csv.DictWriter(f,fieldnames=fields);out.writeheader();f.flush()
            while not stop.is_set() and time.monotonic()-event_started<args.clip_seconds:
                began=time.monotonic()
                if began-last_status_check>=THERMAL_CHECK_INTERVAL_SECONDS:
                    temp,flags=read_pi_status();last_status_check=began
                if finite(temp) and temp>=STOP_TEMPERATURE_C:raise RuntimeError('Thermal limit during capture')
                if flags not in ['unknown','unavailable','throttled=0x0']:raise RuntimeError(f'Power/thermal flags: {flags}')
                if motor.failed.is_set():raise RuntimeError(f'UDP fault: {motor.error}')
                images={};stamps={}
                resources={'top':top_cap,'left':sides['underwater_left'],'right':sides['underwater_right']}
                for role,resource in resources.items():
                    if not resource.grab():raise RuntimeError('Camera grab failed: '+role)
                for role,resource in resources.items():
                    ok,image=resource.retrieve();stamps[role]=time.monotonic()
                    if not ok or image is None:raise RuntimeError('Camera retrieve failed: '+role)
                    images[role]=image
                # Screen-zone control does not infer or require AUV heading.
                heading_now=math.nan;heading_status='screen_zone_no_heading_estimate'
                observation=tracker.update(images['top'],heading)
                now=time.monotonic();current_yaw=motor.snapshot()[0]
                if last_yaw and not current_yaw:
                    next_pulse=max(next_pulse,now+args.settle_seconds)
                request=0.;pulse_requested=False
                valid_track=bool(observation.valid and finite(observation.v_px))
                motor.set_tracking_freshness(valid_track,now,args.track_max_age)
                if not valid_track:
                    motor.cancel('target_invalid')
                    args.zone_policy.update(math.nan,False,False,now)
                    decision='target_invalid'
                elif args.condition=='stationary':
                    motor.cancel('stationary_control');decision='stationary'
                    args.zone_policy.reset_observations()
                else:
                    request,decision=args.zone_policy.update(
                        observation.v_px,valid_track,observation.trigger_ready,now,
                        blocked=bool(current_yaw) or now<next_pulse)
                    if request and pulses>=args.max_pulses_per_event:
                        request=0.;decision='maximum_yaw_count_reached'
                    elif request and not stop.is_set():
                        if motor.pulse(request):
                            args.zone_policy.consume();pulses+=1;pulse_requested=True
                            next_pulse=now+args.pulse_seconds+args.settle_seconds
                            decision='zone_right_pulse' if request>0 else 'zone_left_pulse'
                        else:
                            request=0.;decision='pulse_rejected_'+motor.last_reject
                actual=motor.snapshot()[0];last_yaw=bool(actual)
                phase='yaw_command' if (actual or current_yaw or request) else ('settle_command_zero' if time.monotonic()<next_pulse else 'observation_zero')
                row=dict(index=records,skew_s=max(stamps.values())-min(stamps.values()),phase=phase,sharpness=0.,
                    elapsed_capture_s=time.monotonic()-event_started,tracking_valid=observation.valid,
                    consecutive=observation.stable_frames,fish_x_cm=csv_value(observation.x_cm),fish_y_cm=csv_value(observation.y_cm),
                    bearing_error_deg=csv_value(observation.bearing_error_deg),heading_deg=csv_value(heading_now),
                    heading_status=heading_status,requested_yaw=request,sender_yaw=actual,control_decision=decision,temperature_c=csv_value(temp),
                    tracking_timestamp_monotonic=stamps['top'],fish_u_px=csv_value(observation.u_px),fish_v_px=csv_value(observation.v_px),zone=args.zone_policy.zone,
                    zone_armed=args.zone_policy.armed,zone_stable_count=args.zone_policy.count)
                for role,image in images.items():
                    writers[role].write(image)
                    row[role+'_file']=role+'.avi';row['t_'+role]=stamps[role]
                out.writerow(row);f.flush();records+=1
                stop.wait(max(0,1/args.record_fps-(time.monotonic()-began)))
        elapsed=time.monotonic()-event_started
        if elapsed and records/elapsed<args.record_fps*.9:
            print(f'WARNING: achieved {records/elapsed:.2f} triplets/s, requested {args.record_fps:.2f}. Inspect timestamps.',flush=True)
        return {'status':'captured' if records else 'empty_capture','frames':records,'pulses':pulses,
            'elapsed_capture_s':elapsed,'requested_capture_s':args.clip_seconds,
            'capture_completed_before_deadline':not stop.is_set(),
            'effective_triplet_fps':records/elapsed if elapsed else 0,
            'timestamps':'grab-all then sequential retrieve-completion; not exposure synchronization',
            'requested_camera_fps':25,'recording_format':'MJPG AVI',
            'rate_note':'effective_triplet_fps is measured; AVI nominal25fps is not proof of achieved25fps',
            'physical_measurement_valid':False}
    finally:
        motor.end_recording('capture_end_zero')
        (event/'yaw_summary.json').write_text(json.dumps(motor.usage(),indent=2))
        errors=[]
        for name,resource in list(sides.items())+list(writers.items()):
            try:resource.release()
            except Exception as exc:errors.append(f'{name}: {exc}')
        if errors:(event/'cleanup_errors.json').write_text(json.dumps(errors))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--no-display',action='store_true',help='Always headless')
    p.add_argument('--tracking-only',action='store_true')
    p.add_argument('--disable-inference',action='store_true')
    p.add_argument('--condition',choices=['stationary','tracking'],default='stationary')
    p.add_argument('--run-seconds',type=float,default=180)
    p.add_argument('--clip-seconds',type=float,default=30)
    p.add_argument('--record-fps',type=float,default=25)
    p.add_argument('--model',default=MODEL_PATH)
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--inference-threads',type=int,default=1)
    p.add_argument('--inference-timeout',type=float,default=7200)
    p.add_argument('--max-triplets',type=int,default=0,help='Ignored: ALL acquired frames inferred')
    p.add_argument('--max-skew',type=float,default=.080)
    p.add_argument('--cooldown-seconds',type=float,default=3)
    p.add_argument('--output-root',type=Path,default=Path(OUTPUT_ROOT)/'five_minute_trials')
    p.add_argument('--heading-deg',type=float,default=180)
    p.add_argument('--heading-file',type=Path,help='Existing external MEASURED heading publisher, not supplied by this program')
    p.add_argument('--heading-max-age',type=float,default=.25)
    p.add_argument('--yaw-sign',type=int,choices=[-1,1],default=-1)
    p.add_argument('--yaw-magnitude',type=float,default=.15)
    p.add_argument('--pulse-seconds',type=float,default=1.0)
    p.add_argument('--deadband-deg',type=float,default=8)
    p.add_argument('--settle-seconds',type=float,default=2.0)
    p.add_argument('--max-pulses-per-event','--max-yaw-count',dest='max_pulses_per_event',type=int,default=5,help='Maximum corrections during recording; not a required number; 0 disables yaw requests')
    p.add_argument('--max-total-motor-seconds',type=float,default=None,help='Optional requested-on-time ceiling; default min(max-yaw-count*pulse-seconds, clip-seconds, 30)')
    p.add_argument('--enable-motors',action='store_true')
    p.add_argument('--confirm-restrained',action='store_true')
    p.add_argument('--confirm-watchdog-tested',action='store_true')
    p.add_argument('--confirm-yaw-sign-tested',action='store_true')
    p.add_argument('--confirm-heading-tested',action='store_true')
    p.add_argument('--replay-event',type=Path)
    p.add_argument('--save-event-video',action='store_true',help='Optional MP4; default PNG sequences avoid unresolved release abort')
    p.add_argument('--esp32-ip',default=ESP32_IP)
    p.add_argument('--esp32-port',type=int,default=ESP32_PORT)
    p.add_argument('--zone-center-y',type=float,default=None,help='Neutral reference line in native 640x480 top image; required for powered use')
    p.add_argument('--zone-half-band',type=float,default=20.0)
    p.add_argument('--zone-stable-frames',type=int,default=3)
    p.add_argument('--zone-stable-seconds',type=float,default=.5)
    p.add_argument('--track-max-age',type=float,default=1.5)
    p.add_argument('--confirm-zone-mapping',action='store_true',help='Upper=right and lower=left verified in this fixed camera setup')
    p.add_argument('--confirm-reference-geometry',action='store_true',help='Underwater view/crop matches reference after resizing; live top mapping verified; fixed AUV position and valid reference axis-range model')
    p.add_argument('--useful-policy',choices=['four-keypoints','measurement-pairs'],default='four-keypoints',help='Default preserves frozen rule; measurement-pairs allows length/height independently')
    p.add_argument('--inference-stride',type=int,default=2)
    p.add_argument('--top-association-max-px',type=float,default=100.)
    p.add_argument('--save-annotations',action='store_true',help='Save annotated JPEGs only for inferred frames')
    p.add_argument('--allow-analysis-overrun',action='store_true',help='Finish sampled inference beyond run-seconds; report overrun')
    p.add_argument('--link-only',action='store_true')
    return p.parse_args()


def inference_worker(event_dir,model_path,imgsz,threads,max_triplets,skew_limit,analysis_timeout_override=None,useful_policy_override=None):
    """Preserve inference/selection, then export new CSV results including partial rows."""
    directory=Path(event_dir)
    before={p.resolve() for p in directory.glob('physical_inference_*/measurements.csv')}
    try:
        return _inference_worker_without_npz(event_dir,model_path,imgsz,threads,max_triplets,skew_limit,analysis_timeout_override=analysis_timeout_override,useful_policy_override=useful_policy_override)
    finally:
        for source in sorted(directory.glob('physical_inference_*/measurements.csv')):
            if source.resolve() in before:continue
            try:
                from measurement_npz import export_npz
                output=export_npz(source,source.parent/'measurement_results.npz')
                print(f'NPZ saved: {output}',flush=True)
            except Exception as exc:
                print(f'WARNING: NPZ export failed: {type(exc).__name__}: {exc}. CSV retained; export it separately.',flush=True)


def validate_yaw_options(args):
    """Validate independent controls; maximum yaw count has no five-pulse ceiling."""
    context=(f'pulse_seconds={args.pulse_seconds!r}, '
             f'yaw_magnitude={args.yaw_magnitude!r}, '
             f'max_yaw_count={args.max_pulses_per_event!r}, '
             f'max_total_motor_seconds={args.max_total_motor_seconds!r}')
    if not finite(args.pulse_seconds) or not .05<=args.pulse_seconds<=1.0:
        raise ValueError('--pulse-seconds must be 0.05..1.0 seconds per pulse. '+context)
    if not finite(args.yaw_magnitude) or not 0<args.yaw_magnitude<=.15:
        raise ValueError('--yaw-magnitude must be greater than 0 and at most 0.15. '+context)
    if not isinstance(args.max_pulses_per_event,int) or args.max_pulses_per_event<0:
        raise ValueError('--max-yaw-count/--max-pulses-per-event must be a nonnegative integer. '+context)
    if not finite(args.settle_seconds) or args.settle_seconds<.5:
        raise ValueError(f'--settle-seconds must be at least 0.5; received {args.settle_seconds!r}')
    if args.max_total_motor_seconds is None:
        args.max_total_motor_seconds=min(args.max_pulses_per_event*args.pulse_seconds,args.clip_seconds,30.)
    if not finite(args.max_total_motor_seconds) or not 0<=args.max_total_motor_seconds<=30:
        raise ValueError('--max-total-motor-seconds must be 0..30 seconds; '
                         f'received {args.max_total_motor_seconds!r}')
    return args


def main():
    args=parse_args()
    for name in ['run_seconds','clip_seconds','record_fps','inference_timeout','max_skew','heading_max_age']:
        v=getattr(args,name)
        if not finite(v) or v<=0:raise ValueError(f'Invalid {name}')
    if args.inference_stride<1 or not finite(args.top_association_max_px) or args.top_association_max_px<=0:
        raise ValueError('Invalid stride/association setting')
    if not finite(args.heading_deg) or not finite(args.deadband_deg) or not 1<=args.deadband_deg<=45:
        raise ValueError('Invalid heading/deadband')
    if not finite(args.cooldown_seconds) or args.cooldown_seconds<0:raise ValueError('Invalid cooldown')
    if not 0<args.run_seconds<=300 or not 0<args.clip_seconds<=300 or not 1<=args.record_fps<=25:
        raise ValueError('Trial/capture limits')
    validate_yaw_options(args)
    print(f'Yaw configuration: maximum count={args.max_pulses_per_event}, '
          f'magnitude={args.yaw_magnitude:.3f}, pulse={args.pulse_seconds:.3f}s, '
          f'settling={args.settle_seconds:.3f}s, '
          f'requested-on budget={args.max_total_motor_seconds:.3f}s, '
          f'recording={args.clip_seconds:.3f}s',flush=True)
    if args.imgsz != 1280 or not 1<=args.inference_threads<=4 or not 1<=args.esp32_port<=65535:
        raise ValueError('Invalid inference/network setting')
    if not args.confirm_reference_geometry and not (args.link_only or args.tracking_only or args.disable_inference):
        raise RuntimeError('Confirm underwater reference FOV, live top mapping and fixed-pose axis-range assumptions before cm conversion')
    center=AUV_MASK_CENTER_PIXEL[1] if args.zone_center_y is None else args.zone_center_y
    if (not finite(center) or not finite(args.zone_half_band) or args.zone_half_band<=0
        or not 0<center-args.zone_half_band<center+args.zone_half_band<480
        or args.zone_stable_frames<2 or not finite(args.zone_stable_seconds)
        or args.zone_stable_seconds<.2 or not finite(args.track_max_age) or not .2<=args.track_max_age<=2):
        raise ValueError('Invalid screen-zone or freshness settings')
    if args.replay_event:
        if args.enable_motors or args.link_only:raise RuntimeError('Replay is offline only')
        inference_worker(str(args.replay_event),args.model,args.imgsz,args.inference_threads,0,args.max_skew,analysis_timeout_override=args.inference_timeout,useful_policy_override=args.useful_policy)
        return 0
    if args.link_only and args.enable_motors:raise RuntimeError('Link-only is zero-only')
    if args.enable_motors:
        if args.zone_center_y is None or not args.confirm_zone_mapping:
            raise RuntimeError('Powered zone control requires explicit --zone-center-y and --confirm-zone-mapping')
        if args.heading_file:
            raise RuntimeError('This version uses screen zones, not a heading publisher')
        if args.condition!='tracking' or args.tracking_only or not all([args.confirm_restrained,args.confirm_watchdog_tested,args.confirm_yaw_sign_tested]):
            raise RuntimeError('Powered tracking requires restraint, tested watchdog/sign')
        if not sys.stdin.isatty():raise RuntimeError('Powered trial requires interactive terminal')
        if args.heading_file and not args.confirm_heading_tested:raise RuntimeError('Measured heading must be independently verified')
        if args.heading_file and not read_heading(args.heading_file,args.heading_max_age,args.heading_deg)[1]:
            raise RuntimeError('Measured heading unavailable/stale')
    if not args.disable_inference and not args.tracking_only and not args.link_only:
        if not any(Path(args.model).glob('*.param')) or not any(Path(args.model).glob('*.bin')):raise RuntimeError('Invalid NCNN folder')
    run=args.output_root/(time.strftime('run_%Y%m%d_%H%M%S')+f'_{time.time_ns()%1000000}')
    run.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(control_mode='screen_zone_assistance_not_heading_alignment',
                  zone_center_y_effective=center,zone_mapping='upper=+.15 right; lower=-.15 left',
                  zone_rearm='new stable observations after each pulse and settling; neutral return not required',physical_measurements_enabled=True,
                  yaw_window_policy='tracking-triggered only during recording; neutral ACK before inference',
                  yaw_count_policy='maximum only; never forced to exhaust the budget',trial_deadline_policy='one recording; cooperative analysis budget unless allow-analysis-overrun; partial completion reported',
                  inference='left/right YOLO only; native1280x720; saved tracking centroid for position; configurable useful policy',camera_info=CAMERAS,
                  camera_resolutions=CAMERA_RESOLUTIONS,
                  camera_controls_policy='No application-level image-control overrides',
                  yaw_convention='positive right, negative left; bearing error is counterclockwise-positive',
                  measurement_scope='reference-polynomial cm estimates; underwater-FOV, live-top-map and fixed-axis-range assumptions confirmed by operator')
    (run/'configuration.json').write_text(json.dumps(config,indent=2))
    motor_stop=threading.Event();abort=threading.Event();expired=threading.Event();arm=threading.Event()
    def operator_stop(*_):abort.set();motor_stop.set()
    signal.signal(signal.SIGINT,operator_stop);signal.signal(signal.SIGTERM,operator_stop)
    args.zone_policy=ZoneYawPolicy(center,args.zone_half_band,args.zone_stable_frames,
        args.zone_stable_seconds,args.track_max_age,args.yaw_magnitude)
    def terminal():
        while not abort.is_set() and not expired.is_set():
            readable,_,_=select.select([sys.stdin],[],[],.1)
            if not readable:continue
            text=sys.stdin.readline()
            if not text or text.strip().lower() in ('','q','stop'):operator_stop();return
            if text.strip().lower()=='arm':arm.set();print('Trial armed',flush=True)
            else:print('Commands: arm | q | stop. Yaw direction is automatic.',flush=True)
    def thermal_check():
        temp,flags=read_pi_status()
        if finite(temp) and temp>=STOP_TEMPERATURE_C:raise RuntimeError('Thermal stop')
        if flags not in ['unknown','unavailable','throttled=0x0']:raise RuntimeError(f'Power/thermal stop: {flags}')
    cap=None;motor=None;worker=None;timer=None;events=0;start=None;outcomes=[];fault=None
    try:
        motor=DeadlineMotor(args.enable_motors or args.link_only,motor_stop,args.pulse_seconds,
            endpoint=(args.esp32_ip,args.esp32_port),log_path=run/'udp_link.csv',
            allow_repeat=True,max_total_on=args.max_total_motor_seconds)
        if args.link_only:
            ok=motor.wait_for_link();print(f'Matched zero-command ACK received={ok}',flush=True);return 0 if ok else 2
        if args.enable_motors and not motor.wait_for_link():raise RuntimeError('No matched ACK to zero preflight')
        cap=open_camera('top',CAMERAS['top']);tracker=SimpleFishTracker()
        if args.enable_motors:
            threading.Thread(target=terminal,daemon=True).start()
            print('Type arm to start automatic zone trial. q/Enter/Ctrl+C stops.',flush=True)
            while not arm.wait(.1):
                if abort.is_set():return 0
                if motor.failed.is_set():raise RuntimeError('Sender failed while unarmed')
        else:arm.set();print('PASSIVE: no motor packets. Trial starts after camera initialization.',flush=True)
        start=time.monotonic()
        def trial_expiry():
            expired.set();motor_stop.set();motor.cancel('trial_deadline_zero')
        timer=threading.Timer(args.run_seconds,trial_expiry);timer.start()
        if args.tracking_only:
            with (run/'tracking_only.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['time','valid','consecutive','x_cm','y_cm','bearing'])
                while not motor_stop.is_set():
                    thermal_check();frame,_=read_camera(cap,'top');est=tracker.update(frame,args.heading_deg)
                    w.writerow([time.time(),est.valid,est.stable_frames,est.x_cm,est.y_cm,est.bearing_error_deg]);f.flush()
            return 0
        with (run/'events.csv').open('w',newline='') as ef:
            fields=['event','status','frames','pulses','capture_s','inference_exit','finished_elapsed_s','finished_within_trial']
            ew=csv.DictWriter(ef,fieldnames=fields);ew.writeheader();ef.flush()
            while not motor_stop.is_set():
                thermal_check()
                if motor.failed.is_set():raise RuntimeError(f'Communication fault: {motor.error}')
                events+=1;event=run/f'event_{events:04d}';event.mkdir()
                (event/'analysis_options.json').write_text(json.dumps({
                    'inference_stride':args.inference_stride,'useful_policy':args.useful_policy,'save_annotations':args.save_annotations,
                    'top_association_max_px':args.top_association_max_px,
                    'analysis_deadline_monotonic':time.monotonic()+args.inference_timeout if args.allow_analysis_overrun else start+args.run_seconds}))
                outcome=capture_event(cap,tracker,args,motor,motor_stop,event,args.heading_deg)
                zero_start=time.monotonic();motor.cancel('inference_zero')
                if args.enable_motors and not motor.wait_zero(zero_start):raise RuntimeError('No matching zero ACK after recording')
                outcome['inference_exit']='disabled'
                (event/'event_summary.json').write_text(json.dumps(outcome,indent=2))
                if outcome['frames'] and not args.disable_inference and not abort.is_set():
                    # Trial expiry does not kill the final zero-motor analysis.
                    worker=mp.get_context('spawn').Process(target=inference_worker,args=(str(event),args.model,args.imgsz,args.inference_threads,0,args.max_skew))
                    worker.start();deadline=time.monotonic()+args.inference_timeout
                    if not args.allow_analysis_overrun:deadline=min(deadline,start+args.run_seconds+10)
                    while worker.is_alive() and not abort.is_set() and time.monotonic()<deadline:
                        if motor.failed.is_set():raise RuntimeError(f'Sender failed during inference: {motor.error}')
                        thermal_check();worker.join(timeout=.5)
                    if worker.is_alive():
                        worker.terminate();worker.join(timeout=2)
                        if worker.is_alive():worker.kill();worker.join(timeout=2)
                        outcome['inference_exit']='operator_stop_or_worker_timeout'
                    else:outcome['inference_exit']=worker.exitcode
                    worker.close();worker=None
                    if outcome['inference_exit']!=0:outcome['status']='inference_failed_or_interrupted'
                    else:
                        summaries=sorted(event.glob('physical_inference_*/summary.json'))
                        if summaries:
                            analysis_summary=json.loads(summaries[-1].read_text())
                            outcome['analysis_complete']=analysis_summary['analysis_complete']
                            if not outcome['analysis_complete']:
                                outcome['status']='analysis_partial_time_budget'
                                print('WARNING: analysis incomplete at time budget; partial results retained.',flush=True)
                elapsed=time.monotonic()-start
                outcome.update(finished_elapsed_s=elapsed,finished_within_trial=elapsed<=args.run_seconds)
                (event/'event_summary.json').write_text(json.dumps(outcome,indent=2));outcomes.append(outcome)
                ew.writerow(dict(event=events,status=outcome['status'],frames=outcome['frames'],pulses=outcome.get('pulses',0),
                    capture_s=outcome.get('elapsed_capture_s',0),inference_exit=outcome['inference_exit'],finished_elapsed_s=elapsed,
                    finished_within_trial=outcome['finished_within_trial']));ef.flush()
                if outcome['status']=='inference_failed_or_interrupted':raise RuntimeError('Inference worker failed; automatic continuation prohibited')
                break  # exactly one recording event per trial
        return 2 if any(x.get('status')=='analysis_partial_time_budget' for x in outcomes) else 0
    except Exception as exc:
        fault=f'{type(exc).__name__}: {exc}';operator_stop()
        if motor:motor.cancel('fault_zero')
        traceback.print_exc();(run/'fatal_error.txt').write_text(fault)
        return 1
    finally:
        motor_stop.set()
        if timer:timer.cancel()
        if worker:
            if worker.is_alive():
                worker.terminate();worker.join(timeout=2)
                if worker.is_alive():worker.kill();worker.join(timeout=2)
            worker.close()
        try:
            if motor:motor.close()
        finally:
            if cap:cap.release()
            (run/'trial_summary.json').write_text(json.dumps({'events_completed':len(outcomes),'fault':fault,
                'trial_expired':expired.is_set(),'elapsed_s':None if start is None else time.monotonic()-start,
                'powered':args.enable_motors,'physical_measurements_enabled':True,
                'within_time_budget':None if start is None else time.monotonic()-start<=args.run_seconds,
                'analysis_overrun_allowed':args.allow_analysis_overrun},indent=2))
        print(f'Output: {run}',flush=True)


if __name__=='__main__':
    raise SystemExit(main())
