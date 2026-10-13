import asyncio
import base64
import csv
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None
import websockets
from ultralytics import YOLO

#SIGNALING_SERVER = "ws://192.168.0.74:9000"
SIGNALING_SERVER = "wss://signaling.ehb.be"
BEARER_TOKEN = "LTddk_ptxQX-omdw5B5rfpniA2wB-19KBxFaKuODMzw"
MODELS_DIR = Path("models")
DETECTION_CONFIDENCE = 0.8
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
LATENCY_LOG_THRESHOLD_MS = 200
LATENCY_LOG_EVENT_COUNT = 10
LATENCY_LOG_WINDOW_SEC = 5
CSV_PATH = Path("latency_log.csv")
CSV_HEADER = [
    "timestamp",
    "longitude",
    "latitude",
    "lastlatency",
    "model_path",
    "detection_confidence",
]
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "ehb/pathnavigation/heading"
RECONNECT_DELAY_MIN_SEC = 1
RECONNECT_DELAY_MAX_SEC = 30
WS_PING_INTERVAL_SEC = 10
WS_PING_TIMEOUT_SEC = 10
WS_SEND_TIMEOUT_SEC = 10
FRAME_PROCESSING_TIMEOUT_SEC = 30
WATCHDOG_TIMEOUT_SEC = 60
PROFILE_WINDOW_FRAMES = 20
ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_DISTANCE_CALIBRATION_POINTS = [
    (1.0, 1587.0),
    (0.5, 6200.0),
    (0.3, 16200.0),
    (1.4,750.0),
]
ARUCO_AREA_AT_1M_PX2 = sum(
    area_px2 * (distance_m**2)
    for distance_m, area_px2 in ARUCO_DISTANCE_CALIBRATION_POINTS
) / len(ARUCO_DISTANCE_CALIBRATION_POINTS)


def load_models(models_dir):
    model_paths = sorted(models_dir.glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No .pt models found in {models_dir.resolve()}")

    models_by_name = {}
    ordered_model_names = []

    for model_path in model_paths:
        model_name = model_path.stem
        ordered_model_names.append(model_name)
        models_by_name[model_name] = {
            "path": model_path,
            "model": YOLO(str(model_path), verbose=False),
        }

    return models_by_name, ordered_model_names


MODELS_BY_NAME, MODEL_ORDER = load_models(MODELS_DIR)
DEFAULT_MODEL_NAME = "unrealsim" if "unrealsim" in MODELS_BY_NAME else MODEL_ORDER[0]


def create_aruco_detector():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "ArucoDetector"):
        print(
            "OpenCV ArUcoDetector is not available; "
            "install opencv-contrib-python to enable marker detection."
        )
        return None

    dictionary_id = getattr(aruco, ARUCO_DICTIONARY_NAME, None)
    if dictionary_id is None:
        print(f"Unknown ArUco dictionary: {ARUCO_DICTIONARY_NAME}")
        return None

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, parameters)


ARUCO_DETECTOR = create_aruco_detector()
# One worker: ArucoDetector is not shared between threads that way
ARUCO_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aruco")


def create_mqtt_client():
    if mqtt is None:
        print("paho-mqtt is not installed; MQTT publish is disabled.")
        return None

    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        print(f"Connected to MQTT broker ({MQTT_BROKER}:{MQTT_PORT})")
        return client
    except Exception as exc:
        print(f"MQTT connection failed: {exc}")
        return None


def add_aruco_marker_payload(payload, aruco_markers):
    if not aruco_markers:
        return
    
    payload["aruco_marker_id"] = aruco_markers[0]["id"]
    payload["aruco_marker_area"] = aruco_markers[0]["area_px2"]
    payload["aruco_marker_distance_m"] = aruco_markers[0]["distance_m"]
    payload["aruco_marker_center_x"] = aruco_markers[0]["center_x_px"]
    payload["aruco_marker_center_y"] = aruco_markers[0]["center_y_px"]
    payload["aruco_marker_offset_x"] = aruco_markers[0]["offset_x_px"]
    payload["aruco_marker_horizontal_position"] = aruco_markers[0]["horizontal_position"]
    payload["aruco_markers"] = aruco_markers


def log_aruco_detection(aruco_markers, frame_id, session_id):
    if not aruco_markers:
        return

    marker_summary = ", ".join(
        (
            f"id={marker['id']} "
            f"area={marker['area_px2']}px2"
            f"distance={marker['distance_m']}m "
            f"center=({marker['center_x_px']},{marker['center_y_px']})px "
            f"offset_x={marker['offset_x_px']}px "
            f"position={marker['horizontal_position']}"
        )
        for marker in aruco_markers
    )


def publish_heading(client, heading, session_id, frame_id, aruco_markers=None):
    if client is None:
        return

    payload = {
        "heading": round(heading, 2),
        "sessionId": session_id,
        "frame_id": frame_id,
    }
    add_aruco_marker_payload(payload, aruco_markers)

    try:
        result = client.publish(MQTT_TOPIC, json.dumps(payload))
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"MQTT publish failed with rc={result.rc}")
    except Exception as exc:
        print(f"MQTT publish failed: {exc}")


def resolve_model_name(selected_model):
    if selected_model is None:
        return DEFAULT_MODEL_NAME

    model_key = str(selected_model).strip()
    if not model_key:
        return DEFAULT_MODEL_NAME

    if model_key in MODELS_BY_NAME:
        return model_key

    model_key_no_ext = Path(model_key).stem
    if model_key_no_ext in MODELS_BY_NAME:
        return model_key_no_ext

    if model_key.isdigit():
        model_index = int(model_key) - 1
        if 0 <= model_index < len(MODEL_ORDER):
            return MODEL_ORDER[model_index]

    return DEFAULT_MODEL_NAME


def parse_detection_confidence(payload, fallback):
    if not isinstance(payload, dict):
        return fallback

    raw_value = (
        payload.get("DETECTION_CONFIDENCE")
        if payload.get("DETECTION_CONFIDENCE") is not None
        else payload.get("detection_confidence")
    )
    if raw_value is None:
        raw_value = payload.get("confidence")
    if raw_value is None:
        raw_value = payload.get("conficence")

    if raw_value is None:
        return fallback

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return fallback

    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def parse_latency_ms(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def update_latency_window(event_times, now_monotonic):
    event_times.append(now_monotonic)
    cutoff = now_monotonic - LATENCY_LOG_WINDOW_SEC
    while event_times and event_times[0] < cutoff:
        event_times.popleft()
    return len(event_times) >= LATENCY_LOG_EVENT_COUNT

def ensure_csv_header():
    """Writes the header, or upgrades an old log without timestamp column."""
    try:
        if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
            append_csv_row(CSV_HEADER)
            return

        with CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
            rows = list(csv.reader(csv_file))
        if rows and rows[0] == CSV_HEADER:
            return

        # Old format: prepend an empty timestamp to existing rows. Write to a
        # temp file first so the original survives a failed write.
        tmp_path = CSV_PATH.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            csv_writer.writerow(CSV_HEADER)
            for row in rows[1:]:
                csv_writer.writerow([""] + row)
        os.replace(tmp_path, CSV_PATH)
        print(f"Upgraded {CSV_PATH} with a timestamp column")
    except OSError as exc:
        print(f"Could not prepare {CSV_PATH.resolve()}: {exc}")


def append_csv_row(row, path=CSV_PATH):
    """Appends a row to a csv log; a failing log never stops processing."""
    try:
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file, quoting=csv.QUOTE_ALL).writerow(row)
    except OSError as exc:
        print(f"Could not write to {path.resolve()}: {exc}")


def ensure_timing_csv_header():
    """Writes the header; an existing log with other columns is moved aside."""
    header = timing_csv_header()
    try:
        if TIMING_CSV_PATH.exists() and TIMING_CSV_PATH.stat().st_size > 0:
            with TIMING_CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
                if next(csv.reader(csv_file), None) == header:
                    return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_path = TIMING_CSV_PATH.with_name(
                f"{TIMING_CSV_PATH.stem}_{stamp}{TIMING_CSV_PATH.suffix}"
            )
            os.replace(TIMING_CSV_PATH, old_path)
            print(f"Timing columns changed; previous log moved to {old_path}")
    except OSError as exc:
        print(f"Could not prepare {TIMING_CSV_PATH.resolve()}: {exc}")
        return
    append_csv_row(header, TIMING_CSV_PATH)


def timing_csv_header():
    return ["timestamp", "frames", "fps", "width", "height"] + [
        f"{key}_ms" for key, _ in PROFILE_STAGES
    ]


def decode_message_to_frame(msg):
    """
    msg kan bytes (raw JPEG) of str (JSON met base64 JPEG) zijn.
    Retourneert OpenCV BGR frame of None.
    """
    try:
        if isinstance(msg, (bytes, bytearray)):
            jpeg_bytes = bytes(msg)
        elif isinstance(msg, str):
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                return None

            b64 = payload.get("data")
            if not b64:
                return None
            jpeg_bytes = base64.b64decode(b64)
        else:
            return None

        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_allowed_mask_indices(result, model_names):
    if result.boxes is None or result.boxes.cls is None:
        return []

    allowed_indices = []
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    for index, class_id in enumerate(class_ids):
        label = str(model_names.get(class_id, "")).strip().lower()
        if label in ALLOWED_PATH_LABELS:
            allowed_indices.append(index)
    return allowed_indices


def get_aruco_marker_area(corners):
    points = corners.reshape(4, 2).astype(np.float32)
    area = float(abs(cv2.contourArea(points)))
    return round(area, 2)


def estimate_aruco_marker_distance_m(area_px2):
    if area_px2 <= 0:
        return None
    return round(float(np.sqrt(ARUCO_AREA_AT_1M_PX2 / area_px2)), 2)


def get_horizontal_position_hour(offset_x_px, image_center_x, center_tolerance_px):
    if abs(offset_x_px) <= center_tolerance_px:
        return 12

    max_offset_px = max(image_center_x - center_tolerance_px, 1)
    offset_ratio = min(
        (abs(offset_x_px) - center_tolerance_px) / max_offset_px,
        1.0,
    )

    if offset_x_px < 0:
        if offset_ratio <= 1 / 3:
            return 11
        if offset_ratio <= 2 / 3:
            return 10
        return 9

    if offset_ratio <= 1 / 3:
        return 1
    if offset_ratio <= 2 / 3:
        return 2
    return 3


def detect_aruco_markers(frame, center_tolerance_px=30):
    if ARUCO_DETECTOR is None:
        return []

    h, w = frame.shape[:2]
    image_center_x = w / 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = ARUCO_DETECTOR.detectMarkers(gray)

    if marker_ids is None:
        return []

    markers = []
    marker_ids = marker_ids.flatten().astype(int).tolist()

    for marker_id, corners in zip(marker_ids, marker_corners):
        points = corners.reshape(4, 2).astype(np.float32)

        marker_center_x = float(np.mean(points[:, 0]))
        marker_center_y = float(np.mean(points[:, 1]))

        offset_x_px = marker_center_x - image_center_x

        horizontal_position = get_horizontal_position_hour(
            offset_x_px,
            image_center_x,
            center_tolerance_px,
        )

        area_px2 = get_aruco_marker_area(corners)

        markers.append(
            {
                "id": marker_id,
                "area_px2": area_px2,
                "distance_m": estimate_aruco_marker_distance_m(area_px2),
                "center_x_px": round(marker_center_x, 2),
                "center_y_px": round(marker_center_y, 2),
                "offset_x_px": round(offset_x_px, 2),
                "horizontal_position": horizontal_position,
            }
        )

    return sorted(markers, key=lambda marker: marker["area_px2"], reverse=True)


def compute_heading_to_point(frame, target_x, target_y):
    h, w = frame.shape[:2]
    start_x = w // 2
    start_y = h

    dx = target_x - start_x
    dy = start_y - target_y
    return float(np.degrees(np.arctan2(dy, dx)))


def compute_heading_to_marker(frame, aruco_markers):
    if not aruco_markers:
        return None

    marker = aruco_markers[0]
    return compute_heading_to_point(
        frame,
        marker["center_x_px"],
        marker["center_y_px"],
    )


def compute_heading(frame, model=None, return_masks=False, timings=None):
    if timings is None:
        timings = {}
    h, w = frame.shape[:2]

    model_name = resolve_model_name(model)
    yolo_model = MODELS_BY_NAME[model_name]["model"]
    model_names = getattr(yolo_model, "names", {})

    with timed(timings, "yolo"):
        results = yolo_model(frame, conf=DETECTION_CONFIDENCE, verbose=False)
    for r in results:
        # Ultralytics' own breakdown of the model call (ms)
        speed = getattr(r, "speed", None) or {}
        for speed_key, timing_key in (
            ("preprocess", "yolo_pre"),
            ("inference", "yolo_infer"),
            ("postprocess", "yolo_post"),
        ):
            if speed.get(speed_key) is not None:
                timings[timing_key] = timings.get(timing_key, 0.0) + speed[speed_key]

    midpoints = []
    result_masks = []
    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue

        allowed_mask_indices = get_allowed_mask_indices(r, model_names)
        for mask_index in allowed_mask_indices:
            if mask_index >= len(r.masks.data):
                continue

            with timed(timings, "mask_resize"):
                mask_tensor = r.masks.data[mask_index]
                mask = mask_tensor.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if return_masks:
                with timed(timings, "contours"):
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    contour_points = []
                    for contour in contours:
                        if len(contour) == 0:
                            continue
                        contour_points.append(
                            [
                                [int(point[0][0]), int(point[0][1])]
                                for point in contour
                            ]
                        )
                    if contour_points:
                        result_masks.append(contour_points)

            with timed(timings, "scanlines"):
                for rr in SCAN_HEIGHTS:
                    y = int(h * rr)
                    if y >= h:
                        continue
                    idx = np.where(mask[y, :] > 0)[0]
                    if len(idx) > 0:
                        midpoints.append((int(np.mean(idx)), y))

    if not midpoints:
        return 90.0, result_masks

    avg_x = int(np.mean([p[0] for p in midpoints]))
    target_y = min([p[1] for p in midpoints])

    return compute_heading_to_point(frame, avg_x, target_y), result_masks


def detect_aruco_markers_timed(frame, timings):
    with timed(timings, "aruco"):
        return detect_aruco_markers(frame)


def analyze_frame(frame, model_name, return_masks, timings):
    # ArUco (OpenCV, CPU) runs in parallel with the segmentation (YOLO, GPU);
    # both release the GIL, so the frame costs max() instead of the sum.
    aruco_future = ARUCO_EXECUTOR.submit(detect_aruco_markers_timed, frame, timings)
    # De richting komt altijd uit de segmentatie; return_masks bepaalt enkel of
    # de contouren mee teruggaan. Sloeg deze stap over wanneer er geen maskers
    # gevraagd werden, dan kreeg de client altijd 90.0 graden terug en stuurde
    # de begeleiding nooit bij.
    with timed(timings, "segmentation"):
        heading, result_masks = compute_heading(
            frame, model=model_name, return_masks=return_masks, timings=timings
        )
    if not return_masks:
        result_masks = []
    with timed(timings, "aruco_wait"):
        aruco_markers = aruco_future.result()
    marker_heading = compute_heading_to_marker(frame, aruco_markers)
    return aruco_markers, marker_heading, heading, result_masks


@contextmanager
def timed(timings, key):
    """Adds the elapsed time of the block (ms) to timings[key]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start) * 1000


# Printed in this order; sub-steps are indented under their parent step
PROFILE_STAGES = [
    ("decode", "decode (JSON/base64 + JPEG)"),
    ("worker", "worker thread total"),
    ("aruco", "  aruco detection (parallel)"),
    ("segmentation", "  segmentation total (parallel)"),
    ("aruco_wait", "  wait for aruco after segm."),
    ("yolo", "    yolo model call"),
    ("yolo_pre", "      ultralytics preprocess"),
    ("yolo_infer", "      ultralytics inference"),
    ("yolo_post", "      ultralytics postprocess"),
    ("mask_resize", "    mask to cpu + resize"),
    ("contours", "    contours to points"),
    ("scanlines", "    scanlines / heading"),
    ("thread_overhead", "  thread handoff overhead"),
    ("csv", "latency csv log"),
    ("encode", "response JSON encode"),
    ("send", "websocket send"),
    ("mqtt", "mqtt publish"),
    ("total", "total per frame"),
]


class StageProfiler:
    """Collects per-stage timings and prints the averages every N frames."""

    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.frames = 0
        self.totals = {}
        self.counts = {}
        self.frame_shape = None
        self.started = time.perf_counter()

    def add_frame(self, timings, frame_shape):
        for key, value in timings.items():
            self.totals[key] = self.totals.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1
        self.frames += 1
        self.frame_shape = frame_shape
        if self.frames >= self.window:
            self.report()
            self.reset()

    def report(self):
        elapsed = time.perf_counter() - self.started
        fps = self.frames / elapsed if elapsed > 0 else 0.0
        h, w = self.frame_shape[:2]
        lines = [
            f"[timing] last {self.frames} frames: {fps:.1f} fps, "
            f"{w}x{h}, avg ms per frame:"
        ]
        for key, label in PROFILE_STAGES:
            count = self.counts.get(key, 0)
            if count == 0:
                continue
            avg_ms = self.totals[key] / count
            ran_on = f"  ({count}/{self.frames} frames)" if count < self.frames else ""
            lines.append(f"  {label:<32}{avg_ms:8.1f}{ran_on}")
        print("\n".join(lines), flush=True)

        # Same numbers as one csv row; empty when a step did not run
        row = [
            datetime.now().isoformat(timespec="seconds"),
            self.frames,
            round(fps, 2),
            w,
            h,
        ]
        for key, _ in PROFILE_STAGES:
            count = self.counts.get(key, 0)
            row.append(round(self.totals[key] / count, 2) if count else "")
        append_csv_row(row, TIMING_CSV_PATH)


def restart_process(reason):
    print(f"Restarting process: {reason}", flush=True)
    if os.name == "posix":
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        subprocess.Popen([sys.executable] + sys.argv)
        os._exit(1)


_last_heartbeat = time.monotonic()


async def heartbeat():
    global _last_heartbeat
    while True:
        _last_heartbeat = time.monotonic()
        await asyncio.sleep(1)


def start_watchdog():
    """Restarts the process when the asyncio event loop stops responding."""

    def watch():
        while True:
            time.sleep(5)
            stalled_sec = time.monotonic() - _last_heartbeat
            if stalled_sec > WATCHDOG_TIMEOUT_SEC:
                restart_process(f"event loop blocked for {stalled_sec:.0f}s")

    threading.Thread(target=watch, daemon=True, name="watchdog").start()


async def receive_and_infer():
    ssl_context = ssl.create_default_context()
    mqtt_client = create_mqtt_client()
    print(
        f"Loaded models: {', '.join(MODEL_ORDER)}. Default model: {DEFAULT_MODEL_NAME}"
    )
    ensure_csv_header()

    heartbeat_task = asyncio.create_task(heartbeat())  # noqa: F841 (keep a reference)
    start_watchdog()

    reconnect_delay = RECONNECT_DELAY_MIN_SEC
    while True:
        connected_at = time.monotonic()
        try:
            await serve_connection(ssl_context, mqtt_client)
            print("Connection closed by server.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Connection error: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        # Reset the backoff if the connection was healthy for a while
        if time.monotonic() - connected_at > 60:
            reconnect_delay = RECONNECT_DELAY_MIN_SEC
        print(f"Reconnecting in {reconnect_delay}s...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, RECONNECT_DELAY_MAX_SEC)


async def serve_connection(ssl_context, mqtt_client):
    global DETECTION_CONFIDENCE
    async with websockets.connect(SIGNALING_SERVER,
        ssl=ssl_context,   # Uncomment if using wss://
        origin="http://localhost",
        compression=None,
        ping_interval=WS_PING_INTERVAL_SEC,
        ping_timeout=WS_PING_TIMEOUT_SEC,
        additional_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Authorization": f"Bearer {BEARER_TOKEN}"
        },
    ) as ws:
        print(f"Verbonden met signaling server ({SIGNALING_SERVER})")
        pending_frame_meta = {}
        latency_threshold_events = deque()
        latency_burst_active = False
        profiler = StageProfiler(PROFILE_WINDOW_FRAMES)

        while True:
            msg = await ws.recv()
            frame_started = time.perf_counter()
            timings = {}
            frame_meta = {}

            if isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    if payload.get("type") == "frame_meta":
                        DETECTION_CONFIDENCE = parse_detection_confidence(payload, DETECTION_CONFIDENCE)
                        pending_frame_meta = {
                            "frame_id": payload.get("frame_id"),
                            "longitude": payload.get("longitude"),
                            "latitude": payload.get("latitude"),
                            "lastlatency": payload.get("lastlatency"),
                            "model": payload.get("model"),
                            "sessionId": payload.get("sessionId"),
                            "detection_confidence": DETECTION_CONFIDENCE,
                            "returnMasks": payload.get("returnMasks", False),
                            "sendMQTT": payload.get("sendMQTT", False),
                        }
                        continue
                except json.JSONDecodeError:
                    pass

            with timed(timings, "decode"):
                frame = decode_message_to_frame(msg)

            if isinstance(msg, (bytes, bytearray)):
                frame_meta = pending_frame_meta
                pending_frame_meta = {}
            elif isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    frame_meta = {
                        "frame_id": payload.get("frame_id", pending_frame_meta.get("frame_id")),
                        "longitude": payload.get("longitude", pending_frame_meta.get("longitude")),
                        "latitude": payload.get("latitude", pending_frame_meta.get("latitude")),
                        "lastlatency": payload.get("lastlatency", pending_frame_meta.get("lastlatency")),
                        "model": payload.get("model", pending_frame_meta.get("model")),
                        "sessionId": payload.get("sessionId", pending_frame_meta.get("sessionId")),
                        "detection_confidence": parse_detection_confidence(
                            payload,
                            pending_frame_meta.get("detection_confidence", DETECTION_CONFIDENCE),
                        ),
                        "returnMasks": payload.get(
                            "returnMasks", pending_frame_meta.get("returnMasks", False)
                        ),
                        "sendMQTT": payload.get(
                            "sendMQTT", pending_frame_meta.get("sendMQTT", False)
                        ),
                    }
                except Exception:
                    frame_meta = pending_frame_meta
                pending_frame_meta = {}

            if frame is None:
                continue

            frame_id = frame_meta.get("frame_id")
            longitude = frame_meta.get("longitude")
            latitude = frame_meta.get("latitude")
            lastlatency = frame_meta.get("lastlatency")
            lastmodel = frame_meta.get("model")
            sessionId = frame_meta.get("sessionId")
            DETECTION_CONFIDENCE = frame_meta.get("detection_confidence", DETECTION_CONFIDENCE)
            returnMasks = bool(frame_meta.get("returnMasks", False))
            sendMQTT = bool(frame_meta.get("sendMQTT", False))
            resolved_model_name = resolve_model_name(lastmodel)
            # Run inference in a worker thread so the event loop keeps answering
            # websocket pings; a hung inference is unrecoverable, so restart.
            try:
                with timed(timings, "worker"):
                    aruco_markers, marker_heading, heading, resultMasks = await asyncio.wait_for(
                        asyncio.to_thread(
                            analyze_frame, frame, resolved_model_name, returnMasks, timings
                        ),
                        timeout=FRAME_PROCESSING_TIMEOUT_SEC,
                    )
            except asyncio.TimeoutError:
                restart_process(
                    f"frame processing exceeded {FRAME_PROCESSING_TIMEOUT_SEC}s"
                )
            timings["thread_overhead"] = max(
                timings["worker"]
                - timings.get("segmentation", 0.0)
                - timings.get("aruco_wait", 0.0),
                0.0,
            )
            #log_aruco_detection(aruco_markers, frame_id, sessionId)
            model_path = MODELS_BY_NAME[resolved_model_name]["path"]
            latency_ms = parse_latency_ms(lastlatency)


            should_log_latency = False
            if latency_ms is not None and latency_ms > LATENCY_LOG_THRESHOLD_MS:
                now_monotonic = time.monotonic()
                burst_threshold_met = update_latency_window(
                    latency_threshold_events, now_monotonic
                )
                if burst_threshold_met and not latency_burst_active:
                    should_log_latency = True
                latency_burst_active = burst_threshold_met
            else:
                latency_threshold_events.clear()
                latency_burst_active = False

            if should_log_latency:
                with timed(timings, "csv"):
                    append_csv_row(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            longitude,
                            latitude,
                            latency_ms,
                            str(model_path),
                            DETECTION_CONFIDENCE,
                        ]
                    )

            with timed(timings, "encode"):
                response_payload = {
                    "heading": round(heading, 2),
                    "frame_id": frame_id,
                    "sessionId": sessionId,
                }
                add_aruco_marker_payload(response_payload, aruco_markers)

                if marker_heading is not None:
                    response_payload["marker_heading"] = round(marker_heading, 2)

                # Only include resultMasks if marker_heading is not available, to save bandwidth when possible
                if marker_heading is None:
                    response_payload["resultMasks"] = resultMasks
                response_json = json.dumps(response_payload)


            #print (response_payload)
            #print (f"Model: {resolved_model_name} ({model_path}), "
            #    f"Heading: {response_payload['heading']}°, "
            #    f"Marker Heading: {response_payload.get('marker_heading', 'N/A')}°, "
            #    f"Frame ID: {frame_id}, Session ID: {sessionId}, "
            #    f"ArUco Markers: {len(aruco_markers)}, "
            #    f"Detection Confidence: {DETECTION_CONFIDENCE}, "
            #    f"Latency: {latency_ms}ms"
            #)
            with timed(timings, "send"):
                await asyncio.wait_for(
                    ws.send(response_json), timeout=WS_SEND_TIMEOUT_SEC
                )
            if sendMQTT:
                with timed(timings, "mqtt"):
                    publish_heading(
                        mqtt_client, heading, sessionId, frame_id, aruco_markers
                    )

            timings["total"] = (time.perf_counter() - frame_started) * 1000
            profiler.add_frame(timings, frame.shape)


if __name__ == "__main__":
    asyncio.run(receive_and_infer())
