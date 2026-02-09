import asyncio
import json
import base64
import time

import websockets
import cv2
import numpy as np
from ultralytics import YOLO

# --- Minimale vaste instellingen ---
SIGNALING_SERVER = "ws://192.168.0.74:9000"
MODEL_PATH = r"models\unrealsim.pt"
DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

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
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None

async def receive_and_infer():
    async with websockets.connect(SIGNALING_SERVER, max_size=None) as ws:
        pending_frame_id = None  # als er ooit frame_meta komt, houden we het stil bij

        while True:
            msg = await ws.recv()

            frame_id = None
            frame = None

            # Stilletjes frame_meta negeren (maar frame_id wel meenemen indien aanwezig)
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
                # als het een JSON frame was met frame_id
                try:
                    payload = json.loads(msg)
                    frame_id = payload.get("frame_id", pending_frame_id)
                    pending_frame_id = None
                except Exception:
                    frame_id = pending_frame_id
                    pending_frame_id = None

            if frame is None:
                continue

            h, w = frame.shape[:2]
            overlay = frame.copy()

            # --- Inference ---
            results = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)

            midpoints = []

            for r in results:
                if r.masks is None or len(r.masks.data) == 0:
                    continue

                # neem eerste mask
                mask = r.masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # groen overlay op mask
                green = np.full_like(frame, (0, 255, 0))
                blended = cv2.addWeighted(frame, 0.3, green, 0.7, 0)
                overlay[mask > 0] = blended[mask > 0]

                # scanlijnen + midpoints
                for rr in SCAN_HEIGHTS:
                    y = int(h * rr)
                    if y >= h:
                        continue

                    scan_row = mask[y, :]
                    idx = np.where(scan_row > 0)[0]
                    if len(idx) > 0:
                        mx = int(np.mean(idx))
                        midpoints.append((mx, y))
                        cv2.circle(overlay, (mx, y), 5, (255, 0, 0), -1)

                    cv2.line(overlay, (0, y), (w, y), (150, 150, 150), 1)

            # --- Heading + pijl ---
            direction_angle = 90.0
            start_point = (w // 2, h)

            if midpoints:
                avg_x = int(np.mean([p[0] for p in midpoints]))
                target_point = (avg_x, min([p[1] for p in midpoints]))

                dx = avg_x - start_point[0]
                dy = start_point[1] - target_point[1]
                direction_angle = float(np.degrees(np.arctan2(dy, dx)))

                cv2.arrowedLine(overlay, start_point, target_point, (0, 0, 255), 5, tipLength=0.2)
            else:
                # optioneel: pijl rechtdoor als fallback
                cv2.arrowedLine(overlay, start_point, (w // 2, int(h * 0.6)), (0, 0, 255), 5, tipLength=0.2)

            # Toon resultaat
            cv2.imshow("Segmentation + Heading", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await ws.send(json.dumps({"heading": round(direction_angle, 2), "frame_id": frame_id}))

    cv2.destroyAllWindows()

asyncio.run(receive_and_infer())
