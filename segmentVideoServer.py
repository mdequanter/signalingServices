import asyncio
import base64
import csv
import json
import ssl
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from ultralytics import YOLO

#SIGNALING_SERVER = "ws://192.168.0.74:9000"
SIGNALING_SERVER = "wss://signaling.ehb.be"
MODEL_PATH1 = r"models/unrealsim.pt"
MODEL_PATH2 = r"models/laerbeekbos.pt"
DETECTION_CONFIDENCE = 0.6
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RECORDS_DIR = Path("records")
CSV_PATH = RECORDS_DIR / "inference_log.csv"
SAVE_INTERVAL_SEC = 10.0

model1 = YOLO(MODEL_PATH1, verbose=False)
model2 = YOLO(MODEL_PATH2, verbose=False)

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


def compute_heading(frame, model=1):
    h, w = frame.shape[:2]
    
    if model == "2":
        model = model2
    else:
        model = model1


    results = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)

    midpoints = []
    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue

        mask = r.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for rr in SCAN_HEIGHTS:
            y = int(h * rr)
            if y >= h:
                continue
            idx = np.where(mask[y, :] > 0)[0]
            if len(idx) > 0:
                midpoints.append((int(np.mean(idx)), y))

    if not midpoints:
        return 90.0

    start_x = w // 2
    start_y = h
    avg_x = int(np.mean([p[0] for p in midpoints]))
    target_y = min([p[1] for p in midpoints])

    dx = avg_x - start_x
    dy = start_y - target_y
    return float(np.degrees(np.arctan2(dy, dx)))


async def receive_and_infer():
    ssl_context = ssl.create_default_context()
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    csv_exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists or CSV_PATH.stat().st_size == 0:
            csv_writer.writerow(
                [
                    "Filename",
                    "datetime",
                    "frame_id",
                    "longitude",
                    "latitude",
                    "heading",
                    "MODEL_PATH",
                    "lastlatency",
                    "sessionId",
                ]
            )

    next_save_at = time.time()

    async with websockets.connect(
        SIGNALING_SERVER,
        ssl=ssl_context,
        origin="https://signaling.ehb.be",
        compression=None,
        additional_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        },
    ) as ws:
        print(f"Verbonden met signaling server ({SIGNALING_SERVER})")
        pending_frame_meta = {}

        while True:
            msg = await ws.recv()
            frame_meta = {}

            if isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    if payload.get("type") == "frame_meta":
                        pending_frame_meta = {
                            "frame_id": payload.get("frame_id"),
                            "longitude": payload.get("longitude"),
                            "latitude": payload.get("latitude"),
                            "lastlatency": payload.get("lastlatency"),
                            "model": payload.get("model"),
                            "sessionId": payload.get("sessionId"),
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

            now = time.time()
            saved_this_frame = False
            out_path = None
            if now >= next_save_at:
                ts = time.strftime("%Y%m%d_%H%M%S")
                saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
                fid = "none" if frame_id is None else str(frame_id)
                out_path = RECORDS_DIR / f"frame_{ts}_id_{fid}.jpg"
                cv2.imwrite(str(out_path), frame)
                next_save_at = now + SAVE_INTERVAL_SEC
                saved_this_frame = True

            print (f"Framedata: {frame_meta}")
            #print (f"Last model used: {lastmodel}, latency: {lastlatency} ms, frame_id: {frame_id}, longitude: {longitude}, latitude: {latitude}")

            heading = compute_heading(frame, model=lastmodel)

            if (lastmodel == "2"):
                model_path = MODEL_PATH2
            else:
                model_path = MODEL_PATH1

            if saved_this_frame and out_path is not None:
                with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
                    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                    csv_writer.writerow(
                        [
                            out_path.name,
                            saved_at,
                            frame_id,
                            longitude,
                            latitude,
                            round(heading, 2),
                            model_path,
                            lastlatency,
                            sessionId,
                        ]
                    )

            await ws.send(
                json.dumps(
                    {
                        "heading": round(heading, 2),
                        "frame_id": frame_id,
                        "sessionId": sessionId,
                    }
                )
            )


if __name__ == "__main__":
    asyncio.run(receive_and_infer())
