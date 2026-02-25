import asyncio
import base64
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
MODEL_PATH = r"models/unrealsim.pt"
DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RECORDS_DIR = Path("records")
SAVE_INTERVAL_SEC = 10.0

model = YOLO(MODEL_PATH, verbose=False)


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


def compute_heading(frame):
    h, w = frame.shape[:2]
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
        pending_frame_id = None

        while True:
            msg = await ws.recv()
            frame_id = None

            if isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    if payload.get("type") == "frame_meta":
                        pending_frame_id = payload.get("frame_id")
                        continue
                except json.JSONDecodeError:
                    pass

            frame = decode_message_to_frame(msg)

            if isinstance(msg, (bytes, bytearray)):
                frame_id = pending_frame_id
                pending_frame_id = None
            elif isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    frame_id = payload.get("frame_id", pending_frame_id)
                except Exception:
                    frame_id = pending_frame_id
                pending_frame_id = None

            if frame is None:
                continue

            now = time.time()
            if now >= next_save_at:
                ts = time.strftime("%Y%m%d_%H%M%S")
                fid = "none" if frame_id is None else str(frame_id)
                out_path = RECORDS_DIR / f"frame_{ts}_id_{fid}.jpg"
                cv2.imwrite(str(out_path), frame)
                next_save_at = now + SAVE_INTERVAL_SEC

            heading = compute_heading(frame)
            await ws.send(
                json.dumps(
                    {
                        "heading": round(heading, 2),
                        "frame_id": frame_id,
                    }
                )
            )


if __name__ == "__main__":
    asyncio.run(receive_and_infer())
