import asyncio
import base64
import csv
import json
import ssl
import time
from collections import deque
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
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "ehb/pathnavigation/heading"
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
    print(
        "ArUco marker detected: "
        f"{marker_summary}, frame_id={frame_id}, sessionId={session_id}",
        flush=True,
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


def compute_heading(frame, model=None, return_masks=False):
    h, w = frame.shape[:2]

    model_name = resolve_model_name(model)
    yolo_model = MODELS_BY_NAME[model_name]["model"]
    model_names = getattr(yolo_model, "names", {})

    results = yolo_model(frame, conf=DETECTION_CONFIDENCE, verbose=False)

    midpoints = []
    result_masks = []
    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue

        allowed_mask_indices = get_allowed_mask_indices(r, model_names)
        for mask_index in allowed_mask_indices:
            if mask_index >= len(r.masks.data):
                continue

            mask_tensor = r.masks.data[mask_index]
            mask = mask_tensor.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if return_masks:
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


async def receive_and_infer():
    global DETECTION_CONFIDENCE
    ssl_context = ssl.create_default_context()
    mqtt_client = create_mqtt_client()
    print(
        f"Loaded models: {', '.join(MODEL_ORDER)}. Default model: {DEFAULT_MODEL_NAME}"
    )
    csv_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        if not csv_exists or CSV_PATH.stat().st_size == 0:
            csv_writer.writerow(
                [
                    "longitude",
                    "latitude",
                    "lastlatency",
                    "model_path",
                    "detection_confidence",
                ]
            )

    async with websockets.connect(SIGNALING_SERVER,
        ssl=ssl_context,   # Uncomment if using wss://
        origin="http://localhost",
        compression=None,
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

        while True:
            msg = await ws.recv()
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
            aruco_markers = detect_aruco_markers(frame)
            marker_heading = compute_heading_to_marker(frame, aruco_markers)
            heading = 90.0
            if returnMasks:
                heading, resultMasks = compute_heading(
                    frame, model=resolved_model_name, return_masks=returnMasks
                )
            else:
                resultMasks = []
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
                with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
                    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                    csv_writer.writerow(
                        [
                            longitude,
                            latitude,
                            latency_ms,
                            str(model_path),
                            DETECTION_CONFIDENCE,
                        ]
                    )

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


            #print (response_payload)
            print (f"Model: {resolved_model_name} ({model_path}), "
                f"Heading: {response_payload['heading']}°, "
                f"Marker Heading: {response_payload.get('marker_heading', 'N/A')}°, "
                f"Frame ID: {frame_id}, Session ID: {sessionId}, "
                f"ArUco Markers: {len(aruco_markers)}, "
                f"Detection Confidence: {DETECTION_CONFIDENCE}, "
                f"Latency: {latency_ms}ms"
            )
            await ws.send(json.dumps(response_payload))
            if sendMQTT:
                publish_heading(
                    mqtt_client, heading, sessionId, frame_id, aruco_markers
                )


if __name__ == "__main__":
    asyncio.run(receive_and_infer())
